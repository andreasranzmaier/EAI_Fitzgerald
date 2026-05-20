#pragma once

#include <memory>
#include <string>
#include <vector>

struct DigitPrediction {
  int digit = -1;
  float confidence = 0.0F;
  std::vector<float> probabilities;
};

class TfliteDigitClassifier {
 public:
  explicit TfliteDigitClassifier(const std::string& model_path);
  ~TfliteDigitClassifier();

  TfliteDigitClassifier(const TfliteDigitClassifier&) = delete;
  TfliteDigitClassifier& operator=(const TfliteDigitClassifier&) = delete;
  TfliteDigitClassifier(TfliteDigitClassifier&&) = delete;
  TfliteDigitClassifier& operator=(TfliteDigitClassifier&&) = delete;

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }
  DigitPrediction Predict(const std::vector<float>& normalized_image_28x28);

 private:
  struct Impl;

  bool Load(const std::string& model_path);
  bool CopyInput(const std::vector<float>& normalized_image_28x28);
  std::vector<float> ReadOutput() const;

  std::unique_ptr<Impl> impl_;
  bool ok_ = false;
  std::string error_message_;
};
