#include "bmp_image.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint16_t ReadLe16(std::istream& input) {
  unsigned char bytes[2]{};
  input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t ReadLe32(std::istream& input) {
  unsigned char bytes[4]{};
  input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::int32_t ReadLe32Signed(std::istream& input) {
  return static_cast<std::int32_t>(ReadLe32(input));
}

}  // namespace

BmpImage LoadBmp24(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open BMP file: " + path);
  }

  const auto magic = ReadLe16(input);
  if (magic != 0x4D42U) {
    throw std::runtime_error("Unsupported image format (expected BMP): " + path);
  }

  const std::uint32_t file_size = ReadLe32(input);
  (void)file_size;
  ReadLe16(input);
  ReadLe16(input);
  const std::uint32_t pixel_data_offset = ReadLe32(input);

  const std::uint32_t dib_header_size = ReadLe32(input);
  if (dib_header_size < 40U) {
    throw std::runtime_error("Unsupported BMP DIB header in: " + path);
  }

  const std::int32_t width = ReadLe32Signed(input);
  const std::int32_t height = ReadLe32Signed(input);
  const std::uint16_t planes = ReadLe16(input);
  const std::uint16_t bits_per_pixel = ReadLe16(input);
  const std::uint32_t compression = ReadLe32(input);
  const std::uint32_t image_size = ReadLe32(input);
  (void)image_size;
  ReadLe32Signed(input);
  ReadLe32Signed(input);
  ReadLe32(input);
  ReadLe32(input);

  if (planes != 1U || bits_per_pixel != 24U || compression != 0U) {
    throw std::runtime_error("Only uncompressed 24-bit BMP files are supported: " + path);
  }
  if (width <= 0 || height == 0) {
    throw std::runtime_error("Invalid BMP dimensions in: " + path);
  }

  input.seekg(static_cast<std::streamoff>(pixel_data_offset), std::ios::beg);
  if (!input) {
    throw std::runtime_error("Invalid BMP pixel offset in: " + path);
  }

  const int abs_height = std::abs(height);
  const std::size_t row_stride = static_cast<std::size_t>(((width * 3) + 3) & ~3);
  std::vector<unsigned char> row(row_stride);

  BmpImage image;
  image.width = width;
  image.height = abs_height;
  image.rgb.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(abs_height) * 3U);

  const bool is_bottom_up = height > 0;

  for (int row_index = 0; row_index < abs_height; ++row_index) {
    input.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
    if (!input) {
      throw std::runtime_error("Unexpected end of BMP file: " + path);
    }

    const int destination_y = is_bottom_up ? (abs_height - 1 - row_index) : row_index;
    unsigned char* destination = image.rgb.data() +
                                 static_cast<std::size_t>(destination_y) *
                                     static_cast<std::size_t>(width) * 3U;

    for (int x = 0; x < width; ++x) {
      const unsigned char blue = row[static_cast<std::size_t>(x) * 3U + 0U];
      const unsigned char green = row[static_cast<std::size_t>(x) * 3U + 1U];
      const unsigned char red = row[static_cast<std::size_t>(x) * 3U + 2U];
      destination[static_cast<std::size_t>(x) * 3U + 0U] = red;
      destination[static_cast<std::size_t>(x) * 3U + 1U] = green;
      destination[static_cast<std::size_t>(x) * 3U + 2U] = blue;
    }
  }

  return image;
}

void SavePgm(const std::string& path, int width, int height, const std::vector<float>& pixels) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid PGM dimensions for: " + path);
  }
  if (pixels.size() != static_cast<std::size_t>(width * height)) {
    throw std::runtime_error("PGM pixel buffer has unexpected size for: " + path);
  }

  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Failed to open PGM output file: " + path);
  }

  output << "P5\n" << width << ' ' << height << "\n255\n";
  for (float value : pixels) {
    const float clamped = std::clamp(value, 0.0F, 1.0F);
    const auto byte_value = static_cast<unsigned char>(clamped * 255.0F + 0.5F);
    output.write(reinterpret_cast<const char*>(&byte_value), 1);
  }
  if (!output) {
    throw std::runtime_error("Failed while writing PGM file: " + path);
  }
}
