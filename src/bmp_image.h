#pragma once

#include <string>
#include <vector>

struct BmpImage {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> rgb;
};

BmpImage LoadBmp24(const std::string& path);
void SavePgm(const std::string& path, int width, int height, const std::vector<float>& pixels);
