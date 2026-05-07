#pragma once

#include <string>

struct CaptureOptions {
  std::string output_path = "/tmp/mnist_capture.bmp";
  int timeout_ms = 1200;
  int width = 640;
  int height = 480;
};

bool CaptureStillBmp(const CaptureOptions& options, std::string* backend_used, std::string* error_message);
