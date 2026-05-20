#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sense_hat_display.h"
#include "tflite_gesture_classifier.h"

namespace {

constexpr const char* kModelPath  = "model.tflite";
constexpr const char* kCsvPath    = "test_gesture.csv";
constexpr bool        kShowOnSenseHat = true;

constexpr int kDefaultWarmupRuns    = 20;
constexpr int kDefaultBenchmarkRuns = 100;

// Human-readable label for each gesture class (matches training label_map).
constexpr const char* kGestureLabels[kGestureNumClasses] = {"A", "B", "C", "?"};

enum class ProgramMode {
  kGestureInference,
  kBenchmarkGestureModel,
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
      << "  " << program_name << " --benchmark --model model.tflite --csv test_gesture.csv --runs 100 --warmup 20\n";
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

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    }
    if (arg == "--gesture") {
      options->mode = ProgramMode::kGestureInference;
      continue;
    }
    if (arg == "--benchmark") {
      options->mode = ProgramMode::kBenchmarkGestureModel;
      continue;
    }
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

// Loads one resampled gesture CSV (50 timesteps × 9 features) into a flat float vector.
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
    while (std::getline(sst, cell, ',')) {
      cells.push_back(cell);
    }
    // Columns 0‥8 are the 9 IMU features; any trailing columns (e.g. label) are ignored.
    for (int i = 0; i < kGestureFeatures; ++i) {
      try {
        gesture_input->push_back(std::stof(cells.at(i)));
      } catch (const std::exception&) {
        std::cerr << "Failed to parse CSV value at column " << i << ".\n";
        return false;
      }
    }
  }

  file.close();
  return true;
}

// Runs one gesture inference from a CSV and shows the result on the Sense HAT.
int RunGestureInference(const ProgramOptions& options,
                        TfliteGestureClassifier* classifier) {
  std::vector<float> gesture_input;
  if (!LoadGestureInput(options.csv_path, &gesture_input)) {
    return 1;
  }

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

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < options.benchmark_runs; ++i) {
    prediction = classifier->Predict(gesture_input);
    if (!classifier->ok()) {
      std::cerr << "Inference failed: " << classifier->error_message() << "\n";
      return 1;
    }
  }
  const auto end = std::chrono::steady_clock::now();

  const double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  const double average_ms = total_ms / static_cast<double>(options.benchmark_runs);

  const int idx = prediction.gesture_index;
  const char* label = (idx >= 0 && idx < kGestureNumClasses) ? kGestureLabels[idx] : "?";

  std::cout << "Gesture benchmark mode\n";
  std::cout << "Model: "           << options.model_path   << "\n";
  std::cout << "CSV: "             << options.csv_path     << "\n";
  std::cout << "Warmup runs: "     << options.warmup_runs  << "\n";
  std::cout << "Runs: "            << options.benchmark_runs << "\n";
  std::cout << "Predicted: "       << label                << "\n";
  std::cout << "Confidence: "      << prediction.confidence << "\n";
  std::cout << "Total ms: "        << total_ms             << "\n";
  std::cout << "Average ms: "      << average_ms           << "\n";

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  ProgramOptions options;

  if (!ParseArgs(argc, argv, &options)) {
    return 1;
  }

  TfliteGestureClassifier classifier(options.model_path);
  if (!classifier.ok()) {
    std::cerr << "Failed to load gesture model: "
              << classifier.error_message() << "\n";
    return 1;
  }

  switch (options.mode) {
    case ProgramMode::kGestureInference:
      return RunGestureInference(options, &classifier);

    case ProgramMode::kBenchmarkGestureModel:
      return RunGestureBenchmark(options, &classifier);
  }

  return 1;
}
