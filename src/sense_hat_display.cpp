#include "sense_hat_display.h"

#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace {

constexpr std::uint8_t kDigitPatterns[10][8] = {
    {0b00111100, 0b01100110, 0b01101110, 0b01110110, 0b01100110, 0b01100110, 0b00111100, 0b00000000},
    {0b00011000, 0b00111000, 0b00011000, 0b00011000, 0b00011000, 0b00011000, 0b00111100, 0b00000000},
    {0b00111100, 0b01100110, 0b00000110, 0b00001100, 0b00110000, 0b01100000, 0b01111110, 0b00000000},
    {0b00111100, 0b01100110, 0b00000110, 0b00011100, 0b00000110, 0b01100110, 0b00111100, 0b00000000},
    {0b00001100, 0b00011100, 0b00101100, 0b01001100, 0b01111110, 0b00001100, 0b00011110, 0b00000000},
    {0b01111110, 0b01100000, 0b01111100, 0b00000110, 0b00000110, 0b01100110, 0b00111100, 0b00000000},
    {0b00111100, 0b01100000, 0b01111100, 0b01100110, 0b01100110, 0b01100110, 0b00111100, 0b00000000},
    {0b01111110, 0b01100110, 0b00000110, 0b00001100, 0b00011000, 0b00011000, 0b00011000, 0b00000000},
    {0b00111100, 0b01100110, 0b01100110, 0b00111100, 0b01100110, 0b01100110, 0b00111100, 0b00000000},
    {0b00111100, 0b01100110, 0b01100110, 0b00111110, 0b00000110, 0b00001100, 0b00111000, 0b00000000},
};

constexpr std::uint8_t kErrorPattern[8] = {
    0b10000001,
    0b01000010,
    0b00100100,
    0b00011000,
    0b00011000,
    0b00100100,
    0b01000010,
    0b10000001,
};

std::string Trim(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
    value.pop_back();
  }
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.erase(value.begin());
  }
  return value;
}

std::string FindSenseHatFramebufferPath() {
  for (int i = 0; i < 16; ++i) {
    std::ostringstream sysfs_path;
    sysfs_path << "/sys/class/graphics/fb" << i << "/name";

    std::ifstream input(sysfs_path.str());
    if (!input) {
      continue;
    }

    std::string name;
    std::getline(input, name);
    if (Trim(name) == "RPi-Sense FB") {
      std::ostringstream device_path;
      device_path << "/dev/fb" << i;
      return device_path.str();
    }
  }
  return {};
}

}  // namespace

SenseHatDisplay::SenseHatDisplay() {
  available_ = OpenFramebuffer();
}

SenseHatDisplay::~SenseHatDisplay() {
  Clear();
  if (framebuffer_ != nullptr && mapping_size_ > 0U) {
    munmap(framebuffer_, mapping_size_);
  }
  if (file_descriptor_ >= 0) {
    close(file_descriptor_);
  }
}

bool SenseHatDisplay::OpenFramebuffer() {
  const std::string framebuffer_path = FindSenseHatFramebufferPath();
  if (framebuffer_path.empty()) {
    error_message_ = "Sense HAT framebuffer not found.";
    return false;
  }

  file_descriptor_ = open(framebuffer_path.c_str(), O_RDWR);
  if (file_descriptor_ < 0) {
    error_message_ = "Failed to open " + framebuffer_path + ": " + std::strerror(errno);
    return false;
  }

  fb_fix_screeninfo fix_info{};
  fb_var_screeninfo var_info{};
  if (ioctl(file_descriptor_, FBIOGET_FSCREENINFO, &fix_info) != 0 ||
      ioctl(file_descriptor_, FBIOGET_VSCREENINFO, &var_info) != 0) {
    error_message_ = "Failed to query framebuffer information.";
    return false;
  }

  if (var_info.bits_per_pixel != 16U) {
    error_message_ = "Unexpected Sense HAT framebuffer format (expected RGB565).";
    return false;
  }

  mapping_size_ = static_cast<std::size_t>(fix_info.line_length) * static_cast<std::size_t>(var_info.yres_virtual);
  line_length_ = static_cast<int>(fix_info.line_length);
  framebuffer_ = static_cast<unsigned char*>(
      mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor_, 0));
  if (framebuffer_ == MAP_FAILED) {
    framebuffer_ = nullptr;
    error_message_ = "Failed to mmap the Sense HAT framebuffer.";
    return false;
  }

  return true;
}

std::uint16_t SenseHatDisplay::MakeRgb565(std::uint8_t red, std::uint8_t green, std::uint8_t blue) const {
  const std::uint16_t r = static_cast<std::uint16_t>((red >> 3U) & 0x1FU);
  const std::uint16_t g = static_cast<std::uint16_t>((green >> 2U) & 0x3FU);
  const std::uint16_t b = static_cast<std::uint16_t>((blue >> 3U) & 0x1FU);
  return static_cast<std::uint16_t>((r << 11U) | (g << 5U) | b);
}

void SenseHatDisplay::WritePattern(const std::uint8_t pattern[8], std::uint16_t color) {
  if (!available_ || framebuffer_ == nullptr) {
    return;
  }

  constexpr std::uint16_t background = 0x0000U;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      const bool on = (pattern[y] & (1U << (7 - x))) != 0U;
      auto* pixel = reinterpret_cast<std::uint16_t*>(framebuffer_ + y * line_length_ + x * 2);
      *pixel = on ? color : background;
    }
  }
}

bool SenseHatDisplay::ShowDigit(int digit, float confidence) {
  if (!available_) {
    return false;
  }
  if (digit < 0 || digit > 9) {
    ShowErrorMarker();
    return false;
  }

  const std::uint8_t brightness = static_cast<std::uint8_t>(
      std::clamp(64.0F + confidence * 191.0F, 32.0F, 255.0F));

  std::uint16_t color = 0;
  if (confidence >= 0.80F) {
    color = MakeRgb565(0, brightness, 0);
  } else if (confidence >= 0.55F) {
    color = MakeRgb565(brightness, brightness, 0);
  } else {
    color = MakeRgb565(brightness, 0, 0);
  }

  WritePattern(kDigitPatterns[digit], color);
  return true;
}

void SenseHatDisplay::ShowErrorMarker() {
  if (!available_) {
    return;
  }
  WritePattern(kErrorPattern, MakeRgb565(255, 0, 0));
}

void SenseHatDisplay::Clear() {
  if (!available_ || framebuffer_ == nullptr) {
    return;
  }
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      auto* pixel = reinterpret_cast<std::uint16_t*>(framebuffer_ + y * line_length_ + x * 2);
      *pixel = 0x0000U;
    }
  }
}
