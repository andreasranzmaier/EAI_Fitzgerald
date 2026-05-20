#include "tflite_digit_classifier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace {

// Checks whether the tensor type is supported.
bool IsSupportedTensorType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteInt8;
}

// Converts tensor types to readable names for error messages.
const char* TensorTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return "float32";
    case kTfLiteInt8:
      return "int8";
    default:
      return "unsupported";
  }
}

// Converts a float input value to int8.
int8_t QuantizeInt8(float value, float scale, int zero_point) {
  int quantized = static_cast<int>(
      std::lround(value / scale + static_cast<float>(zero_point)));

  const int min_value =
      static_cast<int>(std::numeric_limits<int8_t>::min());
  const int max_value =
      static_cast<int>(std::numeric_limits<int8_t>::max());

  if (quantized < min_value) {
    quantized = min_value;
  }

  if (quantized > max_value) {
    quantized = max_value;
  }

  return static_cast<int8_t>(quantized);
}

// Converts an int8 output value back to float.
float DequantizeInt8(int8_t value, float scale, int zero_point) {
  return scale * static_cast<float>(static_cast<int>(value) - zero_point);
}

}  // namespace

struct TfliteDigitClassifier::Impl {
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
};

// Creates the classifier and loads the given TensorFlow Lite model.
TfliteDigitClassifier::TfliteDigitClassifier(const std::string& model_path)
    : impl_(std::make_unique<Impl>()) {
  ok_ = Load(model_path);
}

// Releases classifier resources.
TfliteDigitClassifier::~TfliteDigitClassifier() = default;

// Loads the model, creates the interpreter, and prepares tensors.
bool TfliteDigitClassifier::Load(const std::string& model_path) {
  // Load the .tflite model from disk.
  impl_->model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

  if (!impl_->model) {
    error_message_ = "Failed to load model: " + model_path;
    return false;
  }

  // Create the TensorFlow Lite interpreter.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*impl_->model, resolver);

  if (builder(&impl_->interpreter) != kTfLiteOk || !impl_->interpreter) {
    error_message_ = "Failed to create TensorFlow Lite interpreter.";
    return false;
  }

  // Use one CPU thread for predictable inference.
  impl_->interpreter->SetNumThreads(1);

  // Allocate memory for all tensors.
  if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
    error_message_ = "Failed to allocate tensors.";
    return false;
  }

  // Require exactly one input tensor.
  if (impl_->interpreter->inputs().size() != 1) {
    error_message_ = "The model must have exactly one input tensor.";
    return false;
  }

  // Require exactly one output tensor.
  if (impl_->interpreter->outputs().size() != 1) {
    error_message_ = "The model must have exactly one output tensor.";
    return false;
  }

  // Get the input and output tensors.
  const TfLiteTensor* input = impl_->interpreter->input_tensor(0);
  const TfLiteTensor* output = impl_->interpreter->output_tensor(0);

  // Check that the input tensor is float32 or int8.
  if (input == nullptr || !IsSupportedTensorType(input->type)) {
    error_message_ = "The model input must be float32 or int8; got ";
    error_message_ += input == nullptr ? "null" : TensorTypeName(input->type);
    error_message_ += ".";
    return false;
  }

  // Check that the output tensor is float32 or int8.
  if (output == nullptr || !IsSupportedTensorType(output->type)) {
    error_message_ = "The model output must be float32 or int8; got ";
    error_message_ += output == nullptr ? "null" : TensorTypeName(output->type);
    error_message_ += ".";
    return false;
  }

  // Check for the expected input shape [1, 28, 28, 1].
  if (input->dims == nullptr || input->dims->size != 4 ||
      input->dims->data[0] != 1 || input->dims->data[1] != 28 ||
      input->dims->data[2] != 28 || input->dims->data[3] != 1) {
    error_message_ = "Expected input shape [1, 28, 28, 1].";
    return false;
  }

  // Check for the expected output shape [1, 10].
  if (output->dims == nullptr || output->dims->size != 2 ||
      output->dims->data[0] != 1 || output->dims->data[1] != 10) {
    error_message_ = "Expected output shape [1, 10].";
    return false;
  }

  return true;
}

// Copies float input data into the model input tensor.
bool TfliteDigitClassifier::CopyInput(
    const std::vector<float>& normalized_image_28x28) {
  // Check that the input contains exactly 28 * 28 values.
  if (normalized_image_28x28.size() != 28U * 28U) {
    error_message_ = "Expected 28x28 input values.";
    return false;
  }

  // Get the model input tensor.
  TfLiteTensor* input_tensor = impl_->interpreter->input_tensor(0);

  // Copy float values directly for float32 models.
  if (input_tensor->type == kTfLiteFloat32) {
    float* input = impl_->interpreter->typed_input_tensor<float>(0);

    std::copy(normalized_image_28x28.begin(),
              normalized_image_28x28.end(),
              input);

    return true;
  }

  // Quantize float values before copying for int8 models.
  if (input_tensor->type == kTfLiteInt8) {
    // TODO
    int8_t* input = impl_->interpreter->typed_input_tensor<int8_t>(0);

    for (std::size_t index = 0; index < normalized_image_28x28.size(); ++index) {
      input[index] = QuantizeInt8(
        normalized_image_28x28[index],           
        input_tensor->params.scale,
        input_tensor->params.zero_point);
    }

    return true;
  }

  // This should not happen because Load validates the tensor type.
  error_message_ = "Unsupported input tensor type.";
  return false;
}

// Reads the model output tensor and converts probabilities to floats.
std::vector<float> TfliteDigitClassifier::ReadOutput() const {
  // Get the model output tensor.
  const TfLiteTensor* output_tensor = impl_->interpreter->output_tensor(0);

  // Read float values directly for float32 models.
  if (output_tensor->type == kTfLiteFloat32) {
    const float* output = impl_->interpreter->typed_output_tensor<float>(0);
    return std::vector<float>(output, output + 10);
  }

  // Create one output value for each digit class.
  std::vector<float> probabilities(10, 0.0F);

  // Dequantize int8 values before returning them.
  if (output_tensor->type == kTfLiteInt8) {
    // TODO
    const int8_t* output = impl_->interpreter->typed_output_tensor<int8_t>(0);

    // probabilites needs to be changed for the gestures everything else isn't that bad
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
      probabilities[index] = DequantizeInt8(
        output[index],
        output_tensor->params.scale,
        output_tensor->params.zero_point);
    }

    return probabilities;
  }

  // This should not happen because Load validates the tensor type.
  return probabilities;
}

// Runs inference and returns the most likely digit prediction.
DigitPrediction TfliteDigitClassifier::Predict(
    const std::vector<float>& normalized_image_28x28) {
  DigitPrediction prediction;
  prediction.digit = -1;
  prediction.confidence = 0.0F;

  // Return an empty prediction if loading failed.
  if (!ok_) {
    return prediction;
  }

  // Copy the input image into the TensorFlow Lite input tensor.
  if (!CopyInput(normalized_image_28x28)) {
    ok_ = false;
    return prediction;
  }

  // Run inference.
  if (impl_->interpreter->Invoke() != kTfLiteOk) {
    error_message_ = "TensorFlow Lite invocation failed.";
    ok_ = false;
    return prediction;
  }

  // Read the model output as float probabilities.
  prediction.probabilities = ReadOutput();

  // Find the digit with the highest probability.
  const auto best = std::max_element(
      prediction.probabilities.begin(),
      prediction.probabilities.end());

  prediction.digit =
      static_cast<int>(std::distance(prediction.probabilities.begin(), best));
  prediction.confidence = *best;

  return prediction;
}