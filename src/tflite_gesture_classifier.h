// Exercise 04: Part 2 - header file (basically grabbed 1 to 1 from digit_classifier
// only modified the names and added the constanst for the gesture :])
#pragma once

#include <memory>
#include <string>
#include <vector>

// 50 timesteps x 9 features (accel_x/y/z, gyro_x/y/z, mag_x/y/z)
constexpr int kGestureTimesteps = 50;
constexpr int kGestureFeatures = 9;
constexpr int kGestureInputSize = kGestureTimesteps * kGestureFeatures;
constexpr int kGestureNumClasses = 4;

struct GesturePrediction {
  int gesture_index = -1;
  float confidence = 0.0F;
  std::vector<float> scores;
};

class TfliteGestureClassifier {
 public:
  explicit TfliteGestureClassifier(const std::string& model_path);
  ~TfliteGestureClassifier();

  TfliteGestureClassifier(const TfliteGestureClassifier&) = delete;
  TfliteGestureClassifier& operator=(const TfliteGestureClassifier&) = delete;
  TfliteGestureClassifier(TfliteGestureClassifier&&) = delete;
  TfliteGestureClassifier& operator=(TfliteGestureClassifier&&) = delete;

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }
  GesturePrediction Predict(const std::vector<float>& input);

 private:
  struct Impl;

  bool Load(const std::string& model_path);
  bool CopyInput(const std::vector<float>& input);
  std::vector<float> ReadOutput() const;

  std::unique_ptr<Impl> impl_;
  bool ok_ = false;
  std::string error_message_;
};