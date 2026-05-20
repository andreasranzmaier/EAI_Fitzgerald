#pragma once

#include <string>
#include <vector>

#include "bmp_image.h"

struct PreprocessResult {
  bool success = false;
  std::string error_message;
  bool used_dark_foreground = true;
  float threshold = 0.5F;
  int component_area = 0;
  int bbox_x = 0;
  int bbox_y = 0;
  int bbox_width = 0;
  int bbox_height = 0;
  std::vector<float> mnist_input;
};

PreprocessResult PreprocessForMnist(const BmpImage& image);
