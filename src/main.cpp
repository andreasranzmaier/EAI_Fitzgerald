#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "bmp_image.h"
#include "camera_capture.h"
#include "digit_preprocessor.h"
#include "sense_hat_display.h"
#include "tflite_digit_classifier.h"

// E04 Added
#include <fstream>
#include <sstream>
#include "tflite_gesture_classifier.h"

namespace {

constexpr const char* kModelPath = "model.tflite";
constexpr const char* kCapturePath = "/tmp/mnist_capture.bmp";
constexpr const char* kTestImagePath = "test_digit.bmp";

// E04 Added - here you have to change the path to the choosen .csv file (same name as in deploy_to_pi.sh)
constexpr const char* kCsvPath = "test_gesture.csv";

constexpr int kCaptureTimeoutMs = 1200;
constexpr int kCaptureWidth = 640;
constexpr int kCaptureHeight = 480;
constexpr bool kShowOnSenseHat = true;

constexpr int kDefaultWarmupRuns = 20;
constexpr int kDefaultBenchmarkRuns = 1000;

// Selects which use case should run.
enum class ProgramMode {
  kCameraDigitInference,
  kBenchmarkDigitModel,
};

// Stores command line options.
struct ProgramOptions {
  std::string model_path = kModelPath;
  std::string test_image_path = kTestImagePath;

  // E04 Added
  std::string csv_path = kCsvPath;

  ProgramMode mode = ProgramMode::kCameraDigitInference;
  bool show_on_sense_hat = kShowOnSenseHat;
  int warmup_runs = kDefaultWarmupRuns;
  int benchmark_runs = kDefaultBenchmarkRuns;
};

// Prints the command line usage.
void PrintUsage(const char* program_name) {
  std::cerr
      << "Usage:\n"
      << "  " << program_name << " --camera --model model.tflite\n"
      << "  " << program_name << " --benchmark --model model.tflite --test-image test_digit.bmp --runs 1000 --warmup 20\n"
      << "  " << program_name << " --gesture --model gesture_model.tflite\n"
      << "\n"
      << "Examples:\n"
      << "  " << program_name << " --camera --model artifacts/digit_float32.tflite\n"
      << "  " << program_name << " --camera --model artifacts/digit_int8.tflite\n"
      << "  " << program_name << " --benchmark --model artifacts/synapse_pruned.tflite\n"
      << "  " << program_name << " --benchmark --model artifacts/channel_pruned.tflite\n"
      << "  " << program_name << " --benchmark --model artifacts/neuron_pruned.tflite\n";
}

// Parses a positive integer option value.
bool ParsePositiveInt(const std::string& text, int* value) {
  try {
    const int parsed = std::stoi(text);

    if (parsed <= 0) {
      return false;
    }

    *value = parsed;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Parses all command line arguments.
bool ParseArgs(int argc, char** argv, ProgramOptions* options) {
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];

    // E04 Added - parse the new --csv argument
    // reads the next argument as the csv path
    if (arg == "--csv" && index + 1 < argc) {
      options->csv_path = argv[++index];
      continue;
    }

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    }

    if (arg == "--camera") {
      options->mode = ProgramMode::kCameraDigitInference;
      continue;
    }

    if (arg == "--benchmark") {
      options->mode = ProgramMode::kBenchmarkDigitModel;
      continue;
    }

    if (arg == "--model" && index + 1 < argc) {
      options->model_path = argv[++index];
      continue;
    }

    if (arg == "--test-image" && index + 1 < argc) {
      options->test_image_path = argv[++index];
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

    // Keep the old workflow: ./program model.tflite
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

// Loads and preprocesses one digit image.
bool LoadDigitInput(const std::string& image_path,
                    std::vector<float>* mnist_input) {
  BmpImage image;

  try {
    image = LoadBmp24(image_path);
  } catch (const std::exception& ex) {
    std::cerr << "Failed to load image: " << ex.what() << "\n";
    return false;
  }

  PreprocessResult preprocess = PreprocessForMnist(image);

  if (!preprocess.success) {
    std::cerr << "Preprocessing failed: " << preprocess.error_message << "\n";
    return false;
  }

  *mnist_input = preprocess.mnist_input;
  return true;
}

// E04 Added - Loads and preprocesses one gesture csv file.
bool LoadGestureInput(const std::string& csv_path,
                         std::vector<float>* gesture_input) {
  std::ifstream file(csv_path);

  if (!file.is_open()) {
    std::cerr << "Failed to open CSV file: " << csv_path << "\n";
    return false;
  }

  // Read and discard the header row
  std::string header;
  std::getline(file, header);

  // Defining the feature names, to match csv order
  const std::vector<std::string> kFeatures = {
    "accel_x", "accel_y", "accel_z",
    "gyro_x",  "gyro_y",  "gyro_z",
    "mag_x",   "mag_y",   "mag_z"
  };

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream sst(line);
    std::string cell;

    std::vector<std::string> cells;
    // splits the row by commas and stores the results in a vector of strings
    while (std::getline(sst, cell, ',')) {
      cells.push_back(cell);
    }
    // Parse only the 9 feature columns from the timestep row
    for (int i = 0; i < kGestureFeatures; ++i) {
      try {
        gesture_input->push_back(std::stof(cells.at(i)));
      } catch (const std::exception&) {
        std::cerr << "Failed to parse CSV value.\n";
        return false;
      }
    }
  }

  file.close();
  return true;
}

// Benchmarks one digit model.
int RunDigitBenchmark(const ProgramOptions& options,
                      TfliteDigitClassifier* classifier) {
  std::vector<float> mnist_input;

  if (!LoadDigitInput(options.test_image_path, &mnist_input)) {
    return 1;
  }

  DigitPrediction prediction;

  // Run warmup inferences before measuring.
  for (int index = 0; index < options.warmup_runs; ++index) {
    prediction = classifier->Predict(mnist_input);

    if (!classifier->ok()) {
      std::cerr << "Warmup inference failed: "
                << classifier->error_message() << "\n";
      return 1;
    }
  }

  // Measure repeated inference runtime.
  const auto start = std::chrono::steady_clock::now();

  for (int index = 0; index < options.benchmark_runs; ++index) {
    prediction = classifier->Predict(mnist_input);

    if (!classifier->ok()) {
      std::cerr << "Inference failed: "
                << classifier->error_message() << "\n";
      return 1;
    }
  }

  const auto end = std::chrono::steady_clock::now();

  const double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  const double average_ms =
      total_ms / static_cast<double>(options.benchmark_runs);

  std::cout << "Digit benchmark mode\n";
  std::cout << "Model: " << options.model_path << "\n";
  std::cout << "Test image: " << options.test_image_path << "\n";
  std::cout << "Warmup runs: " << options.warmup_runs << "\n";
  std::cout << "Runs: " << options.benchmark_runs << "\n";
  std::cout << "Predicted digit: " << prediction.digit << "\n";
  std::cout << "Confidence: " << prediction.confidence << "\n";
  std::cout << "Total inference ms: " << total_ms << "\n";
  std::cout << "Average inference ms: " << average_ms << "\n";

  return 0;
}

// Runs digit inference from the camera.
int RunCameraDigitInference(const ProgramOptions& options,
                            TfliteDigitClassifier* classifier) {
  SenseHatDisplay display;

  CaptureOptions capture_options;
  capture_options.output_path = kCapturePath;
  capture_options.timeout_ms = kCaptureTimeoutMs;
  capture_options.width = kCaptureWidth;
  capture_options.height = kCaptureHeight;

  std::string camera_backend;
  std::string capture_error;

  while (true) {
    // Capture one image from the camera.
    if (!CaptureStillBmp(capture_options, &camera_backend, &capture_error)) {
      std::cerr << "Camera capture failed: " << capture_error << "\n";
      return 1;
    }

    std::cout << "Captured image with " << camera_backend << "\n";

    std::vector<float> mnist_input;

    // Convert the captured image into MNIST input values.
    if (!LoadDigitInput(kCapturePath, &mnist_input)) {
      return 1;
    }

    // Run digit prediction.
    DigitPrediction prediction = classifier->Predict(mnist_input);

    if (!classifier->ok()) {
      std::cerr << "Inference failed: "
                << classifier->error_message() << "\n";
      return 1;
    }

    std::cout << "Predicted digit: " << prediction.digit << "\n";
    std::cout << "Confidence: " << prediction.confidence << "\n";

    // Show the predicted digit on the Sense HAT.
    if (options.show_on_sense_hat) {
      if (display.available()) {
        display.ShowDigit(prediction.digit, prediction.confidence);
      } else {
        std::cerr << "Sense HAT unavailable: "
                  << display.error_message() << "\n";
      }
    }
  }
}


}  // namespace

int main(int argc, char** argv) {
  ProgramOptions options;

  // Parse command line options.
  if (!ParseArgs(argc, argv, &options)) {
    return 1;
  }

  switch (options.mode) {
    case ProgramMode::kCameraDigitInference: {
      // Load the digit model, which may be normal, pruned, float32, or int8.
      TfliteDigitClassifier classifier(options.model_path);

      if (!classifier.ok()) {
        std::cerr << "Failed to load digit model: "
                  << classifier.error_message() << "\n";
        return 1;
      }

      return RunCameraDigitInference(options, &classifier);
    }

    case ProgramMode::kBenchmarkDigitModel: {
      // Load the digit model, which may be normal, pruned, float32, or int8.
      TfliteDigitClassifier classifier(options.model_path);

      if (!classifier.ok()) {
        std::cerr << "Failed to load digit model: "
                  << classifier.error_message() << "\n";
        return 1;
      }

      return RunDigitBenchmark(options, &classifier);
    }

  }

  return 1;
}