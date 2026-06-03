#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "sense_hat_display.h"
#include "sense_hat_imu.h"
#include "tflite_gesture_classifier.h"

namespace {

constexpr const char* kModelPath = "model.tflite";
constexpr const char* kCsvPath   = "test_gesture.csv";
constexpr bool kShowOnSenseHat   = true;

constexpr int kDefaultWarmupRuns    = 20;
constexpr int kDefaultBenchmarkRuns = 100;

// Sliding window parameters for stream mode.
// ~60 ms/sample should cover ~5 s of motion.
constexpr int kWindowRawSamples  = 80;  // raw samples kept in the ring buffer
constexpr int kWindowStride      = 20;  // advance this many samples between inferences
constexpr int kConsensusRequired = 3;   // consecutive same-class windows to confirm gesture
constexpr int kPollIntervalMs    = 60;  // IMU poll rate (matches training data recording rate)

constexpr int kGarbageClass = 3;

constexpr const char* kGestureLabels[kGestureNumClasses] = {"A", "B", "C", "?"};

// shutdown for stream mode.
std::atomic<bool> g_shutdown{false};

void HandleSignal(int) {
  g_shutdown = true;
}

// Program options.
enum class ProgramMode {
  kGestureInference,
  kBenchmark,
  kStream,
};

struct ProgramOptions {
  std::string model_path   = kModelPath;
  std::string csv_path     = kCsvPath;
  ProgramMode mode         = ProgramMode::kGestureInference;
  bool show_on_sense_hat   = kShowOnSenseHat;
  int  warmup_runs         = kDefaultWarmupRuns;
  int  benchmark_runs      = kDefaultBenchmarkRuns;
};

void PrintUsage(const char* program_name) {
  std::cerr
      << "Usage:\n"
      << "  " << program_name << " --gesture --model model.tflite --csv test_gesture.csv\n"
      << "  " << program_name << " --benchmark --model model.tflite --csv test_gesture.csv --runs 100 --warmup 20\n"
      << "  " << program_name << " --stream --model model.tflite\n";
}

bool ParsePositiveInt(const std::string& text, int* value) {
  try {
    const int parsed = std::stoi(text);
    if (parsed <= 0) return false;
    *value = parsed;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool ParseArgs(int argc, char** argv, ProgramOptions* options) {
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];

    if (arg == "--help" || arg == "-h") { PrintUsage(argv[0]); return false; }
    if (arg == "--gesture")   { options->mode = ProgramMode::kGestureInference; continue; }
    if (arg == "--benchmark") { options->mode = ProgramMode::kBenchmark;        continue; }
    if (arg == "--stream")    { options->mode = ProgramMode::kStream;            continue; }
    if (arg == "--no-sensehat") { options->show_on_sense_hat = false;            continue; }

    if (arg == "--model" && index + 1 < argc) {
      options->model_path = argv[++index];
      continue;
    }
    if (arg == "--csv" && index + 1 < argc) {
      options->csv_path = argv[++index];
      continue;
    }
    if (arg == "--runs" && index + 1 < argc) {
      if (!ParsePositiveInt(argv[++index], &options->benchmark_runs)) {
        std::cerr << "Invalid --runs value.\n";
        return false;
      }
      continue;
    }
    if (arg == "--warmup" && index + 1 < argc) {
      if (!ParsePositiveInt(argv[++index], &options->warmup_runs)) {
        std::cerr << "Invalid --warmup value.\n";
        return false;
      }
      continue;
    }
    if (arg == "--no-sensehat") {
      options->show_on_sense_hat = false;
      continue;
    }
    // Positional: treat bare argument as model path for backwards compat.
    if (!arg.empty() && arg[0] != '-') {
      options->model_path = arg;
      continue;
    }

    std::cerr << "Unknown or incomplete argument: " << arg << "\n";
    PrintUsage(argv[0]);
    return false;
  }
  return true;
}

// CSV input (single static gesture, used by --gesture and --benchmark).
bool LoadGestureInput(const std::string& csv_path,
                      std::vector<float>* gesture_input) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open CSV file: " << csv_path << "\n";
    return false;
  }

  // Discard header row.
  std::string header;
  std::getline(file, header);

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream sst(line);
    std::string cell;
    std::vector<std::string> cells;
    while (std::getline(sst, cell, ',')) cells.push_back(cell);

    // Columns: timestamp_ms, label, accel_x/y/z, gyro_x/y/z, mag_x/y/z
    // Skip the first two (timestamp + label); read the next 9 as features.
    constexpr int kSkip = 2;
    for (int i = kSkip; i < kSkip + kGestureFeatures; ++i) {
      try {
        gesture_input->push_back(std::stof(cells.at(static_cast<std::size_t>(i))));
      } catch (const std::exception&) {
        std::cerr << "Failed to parse CSV value at column " << i << ".\n";
        return false;
      }
    }
  }

  file.close();
  return true;
}

// Resampling: N raw ImuSamples -> exactly kGestureTimesteps × kGestureFeatures
// via linear interpolation (mirrors the Python resampler used during training).

std::vector<float> ResampleWindow(const std::deque<ImuSample>& buf) {
  const int n = static_cast<int>(buf.size());
  std::vector<float> result;
  result.reserve(kGestureTimesteps * kGestureFeatures);

  for (int j = 0; j < kGestureTimesteps; ++j) {
    const float t   = static_cast<float>(j) / static_cast<float>(kGestureTimesteps - 1);
    const float pos = t * static_cast<float>(n - 1);
    const int   i0  = std::min(static_cast<int>(pos), n - 2);
    const int   i1  = i0 + 1;
    const float a   = pos - static_cast<float>(i0);

    const ImuSample& s0 = buf[static_cast<std::size_t>(i0)];
    const ImuSample& s1 = buf[static_cast<std::size_t>(i1)];

    result.push_back(s0.accel_x + a * (s1.accel_x - s0.accel_x));
    result.push_back(s0.accel_y + a * (s1.accel_y - s0.accel_y));
    result.push_back(s0.accel_z + a * (s1.accel_z - s0.accel_z));
    result.push_back(s0.gyro_x  + a * (s1.gyro_x  - s0.gyro_x));
    result.push_back(s0.gyro_y  + a * (s1.gyro_y  - s0.gyro_y));
    result.push_back(s0.gyro_z  + a * (s1.gyro_z  - s0.gyro_z));
    result.push_back(s0.mag_x   + a * (s1.mag_x   - s0.mag_x));
    result.push_back(s0.mag_y   + a * (s1.mag_y   - s0.mag_y));
    result.push_back(s0.mag_z   + a * (s1.mag_z   - s0.mag_z));
  }
  return result;
}

// Modes
int RunGestureInference(const ProgramOptions& options,
                        TfliteGestureClassifier* classifier) {
  std::vector<float> gesture_input;
  if (!LoadGestureInput(options.csv_path, &gesture_input)) return 1;

  const GesturePrediction prediction = classifier->Predict(gesture_input);
  if (!classifier->ok()) {
    std::cerr << "Inference failed: " << classifier->error_message() << "\n";
    return 1;
  }

  const int idx = prediction.gesture_index;
  const char* label = (idx >= 0 && idx < kGestureNumClasses) ? kGestureLabels[idx] : "?";

  std::cout << "Predicted gesture: " << label << "\n";
  std::cout << "Confidence: " << prediction.confidence << "\n";
  for (int i = 0; i < kGestureNumClasses; ++i) {
    std::cout << "  " << kGestureLabels[i] << ": " << prediction.scores[i] << "\n";
  }

  if (options.show_on_sense_hat) {
    SenseHatDisplay display;
    if (display.available()) {
      display.ShowGesture(idx, prediction.confidence);
    } else {
      std::cerr << "Sense HAT unavailable: " << display.error_message() << "\n";
    }
  }

  return 0;
}

// Benchmarks repeated inference on a single CSV sample to measure latency.
int RunGestureBenchmark(const ProgramOptions& options,
                        TfliteGestureClassifier* classifier) {
  std::vector<float> gesture_input;
  if (!LoadGestureInput(options.csv_path, &gesture_input)) {
    return 1;
  }

  GesturePrediction prediction;

  for (int i = 0; i < options.warmup_runs; ++i) {
    prediction = classifier->Predict(gesture_input);
    if (!classifier->ok()) {
      std::cerr << "Warmup inference failed: " << classifier->error_message() << "\n";
      return 1;
    }
  }

  const auto t_start = std::chrono::steady_clock::now();
  for (int i = 0; i < options.benchmark_runs; ++i) {
    prediction = classifier->Predict(gesture_input);
    if (!classifier->ok()) {
      std::cerr << "Inference failed: " << classifier->error_message() << "\n";
      return 1;
    }
  }
  const auto t_end = std::chrono::steady_clock::now();

  const double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  const double avg_ms = total_ms / static_cast<double>(options.benchmark_runs);

  const int idx = prediction.gesture_index;
  const char* label = (idx >= 0 && idx < kGestureNumClasses) ? kGestureLabels[idx] : "?";

  // Output format matches parse_perf_results.py patterns.
  std::cout << "Gesture benchmark mode\n";
  std::cout << "Model: "                     << options.model_path      << "\n";
  std::cout << "CSV: "                        << options.csv_path        << "\n";
  std::cout << "Warmup runs: "               << options.warmup_runs     << "\n";
  std::cout << "Runs: "                       << options.benchmark_runs  << "\n";
  std::cout << "Predicted gesture: "          << label                   << "\n";
  std::cout << "Predicted digit: "            << idx                     << "\n";
  std::cout << "Confidence: "                 << prediction.confidence   << "\n";
  std::cout << "Total measured inference ms: " << total_ms               << "\n";
  std::cout << "Average inference ms: "        << avg_ms                 << "\n";

  return 0;
}

// Reads from the IMU in a loop, maintains a sliding window, and runs inference
// every kWindowStride new samples once the ring buffer is full.
// A gesture is displayed only after kConsensusRequired consecutive windows
// agree on the same non-garbage class.
int RunStreamMode(const ProgramOptions& options,
                  TfliteGestureClassifier* classifier) {
  SenseHatImu imu;
  if (!imu.available()) {
    std::cerr << "IMU unavailable: " << imu.error_message() << "\n";
    return 1;
  }

  SenseHatDisplay display;
  if (options.show_on_sense_hat && !display.available())
    std::cerr << "Sense HAT display unavailable: " << display.error_message() << "\n";

  std::deque<ImuSample> buffer;
  int samples_since_inference = 0;
  int prev_class              = -1;
  int consensus_count         = 0;

  std::cout << "Stream mode - press Ctrl+C to stop\n";

  while (!g_shutdown) {
    ImuSample sample;
    if (!imu.Read(&sample)) {
      std::cerr << "IMU read error, stopping.\n";
      break;
    }

    buffer.push_back(sample);
    if (static_cast<int>(buffer.size()) > kWindowRawSamples)
      buffer.pop_front();
    ++samples_since_inference;

    // Run inference when the ring buffer is full and stride has elapsed.
    if (static_cast<int>(buffer.size()) >= kWindowRawSamples &&
        samples_since_inference >= kWindowStride) {
      samples_since_inference = 0;

      const std::vector<float> input = ResampleWindow(buffer);
      const GesturePrediction pred   = classifier->Predict(input);
      if (!classifier->ok()) {
        std::cerr << "Inference failed: " << classifier->error_message() << "\n";
        break;
      }

      const int cls = pred.gesture_index;
      const char* lbl = (cls >= 0 && cls < kGestureNumClasses) ? kGestureLabels[cls] : "?";
      std::cout << "Window: " << lbl << " (" << pred.confidence << ")\n";

      if (cls == kGarbageClass) {
        // No gesture in motion - clear display and reset consensus.
        prev_class     = -1;
        consensus_count = 0;
        if (options.show_on_sense_hat && display.available())
          display.Clear();
      } else if (cls == prev_class) {
        ++consensus_count;
        if (consensus_count >= kConsensusRequired &&
            options.show_on_sense_hat && display.available())
          display.ShowGesture(cls, pred.confidence);
      } else {
        prev_class      = cls;
        consensus_count = 1;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
  }

  if (options.show_on_sense_hat && display.available())
    display.Clear();

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  ProgramOptions options;
  if (!ParseArgs(argc, argv, &options)) return 1;

  std::signal(SIGINT,  HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  TfliteGestureClassifier classifier(options.model_path);
  if (!classifier.ok()) {
    std::cerr << "Failed to load gesture model: "
              << classifier.error_message() << "\n";
    return 1;
  }

  switch (options.mode) {
    case ProgramMode::kGestureInference:
      return RunGestureInference(options, &classifier);
    case ProgramMode::kBenchmark:
      return RunGestureBenchmark(options, &classifier);
    case ProgramMode::kStream:
      return RunStreamMode(options, &classifier);
  }
  return 1;
}
