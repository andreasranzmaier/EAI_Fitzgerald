#include "digit_preprocessor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ComponentSelection {
  bool found = false;
  bool touches_border = false;
  int area = 0;
  int min_x = 0;
  int min_y = 0;
  int max_x = 0;
  int max_y = 0;
  std::vector<int> indices;
};

std::vector<float> ToGrayscale(const BmpImage& image) {
  std::vector<float> grayscale(static_cast<std::size_t>(image.width * image.height), 0.0F);
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      const std::size_t offset = static_cast<std::size_t>(y * image.width + x) * 3U;
      const float red = static_cast<float>(image.rgb[offset + 0U]) / 255.0F;
      const float green = static_cast<float>(image.rgb[offset + 1U]) / 255.0F;
      const float blue = static_cast<float>(image.rgb[offset + 2U]) / 255.0F;
      grayscale[static_cast<std::size_t>(y * image.width + x)] =
          0.299F * red + 0.587F * green + 0.114F * blue;
    }
  }
  return grayscale;
}

std::vector<float> GaussianBlur3x3(const std::vector<float>& input, int width, int height) {
  std::vector<float> output(static_cast<std::size_t>(width * height), 0.0F);
  const int kernel[3][3] = {
      {1, 2, 1},
      {2, 4, 2},
      {1, 2, 1},
  };

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0F;
      int weight_sum = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          const int sample_x = std::clamp(x + kx, 0, width - 1);
          const int sample_y = std::clamp(y + ky, 0, height - 1);
          const int weight = kernel[ky + 1][kx + 1];
          sum += input[static_cast<std::size_t>(sample_y * width + sample_x)] * static_cast<float>(weight);
          weight_sum += weight;
        }
      }
      output[static_cast<std::size_t>(y * width + x)] = sum / static_cast<float>(weight_sum);
    }
  }

  return output;
}

float ComputeOtsuThreshold(const std::vector<float>& grayscale) {
  std::array<int, 256> histogram{};
  for (float value : grayscale) {
    const int bin = std::clamp(static_cast<int>(value * 255.0F + 0.5F), 0, 255);
    histogram[static_cast<std::size_t>(bin)] += 1;
  }

  const int total = static_cast<int>(grayscale.size());
  double sum = 0.0;
  for (int i = 0; i < 256; ++i) {
    sum += static_cast<double>(i * histogram[static_cast<std::size_t>(i)]);
  }

  int weight_background = 0;
  double sum_background = 0.0;
  double max_variance = -1.0;
  int threshold = 127;

  for (int i = 0; i < 256; ++i) {
    weight_background += histogram[static_cast<std::size_t>(i)];
    if (weight_background == 0) {
      continue;
    }

    const int weight_foreground = total - weight_background;
    if (weight_foreground == 0) {
      break;
    }

    sum_background += static_cast<double>(i * histogram[static_cast<std::size_t>(i)]);

    const double mean_background = sum_background / static_cast<double>(weight_background);
    const double mean_foreground = (sum - sum_background) / static_cast<double>(weight_foreground);
    const double between_class_variance = static_cast<double>(weight_background) *
                                          static_cast<double>(weight_foreground) *
                                          (mean_background - mean_foreground) *
                                          (mean_background - mean_foreground);

    if (between_class_variance > max_variance) {
      max_variance = between_class_variance;
      threshold = i;
    }
  }

  return static_cast<float>(threshold) / 255.0F;
}

std::vector<std::uint8_t> MakeForegroundMask(
    const std::vector<float>& grayscale,
    float threshold,
    bool foreground_is_dark,
    int* foreground_pixels) {
  std::vector<std::uint8_t> mask(grayscale.size(), 0U);
  int count = 0;
  for (std::size_t i = 0; i < grayscale.size(); ++i) {
    const bool is_foreground = foreground_is_dark ? (grayscale[i] <= threshold) : (grayscale[i] >= threshold);
    mask[i] = static_cast<std::uint8_t>(is_foreground ? 1 : 0);
    count += static_cast<int>(mask[i]);
  }
  *foreground_pixels = count;
  return mask;
}

ComponentSelection SelectLargestComponent(const std::vector<std::uint8_t>& mask, int width, int height) {
  const int pixel_count = width * height;
  std::vector<std::uint8_t> visited(static_cast<std::size_t>(pixel_count), 0U);

  ComponentSelection best_non_border;
  ComponentSelection best_any;

  constexpr int min_component_area = 40;
  const int neighbor_offsets[8][2] = {
      {-1, -1}, {0, -1}, {1, -1},
      {-1, 0},           {1, 0},
      {-1, 1},  {0, 1},  {1, 1},
  };

  for (int start = 0; start < pixel_count; ++start) {
    if (mask[static_cast<std::size_t>(start)] == 0U || visited[static_cast<std::size_t>(start)] != 0U) {
      continue;
    }

    std::deque<int> queue;
    std::vector<int> component_indices;
    queue.push_back(start);
    visited[static_cast<std::size_t>(start)] = 1U;

    int min_x = width;
    int min_y = height;
    int max_x = 0;
    int max_y = 0;
    bool touches_border = false;

    while (!queue.empty()) {
      const int index = queue.front();
      queue.pop_front();
      component_indices.push_back(index);

      const int x = index % width;
      const int y = index / width;
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);
      touches_border = touches_border || x == 0 || y == 0 || x == width - 1 || y == height - 1;

      for (const auto& offset : neighbor_offsets) {
        const int nx = x + offset[0];
        const int ny = y + offset[1];
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
          continue;
        }
        const int neighbor = ny * width + nx;
        if (mask[static_cast<std::size_t>(neighbor)] == 0U || visited[static_cast<std::size_t>(neighbor)] != 0U) {
          continue;
        }
        visited[static_cast<std::size_t>(neighbor)] = 1U;
        queue.push_back(neighbor);
      }
    }

    if (static_cast<int>(component_indices.size()) < min_component_area) {
      continue;
    }

    ComponentSelection candidate;
    candidate.found = true;
    candidate.touches_border = touches_border;
    candidate.area = static_cast<int>(component_indices.size());
    candidate.min_x = min_x;
    candidate.min_y = min_y;
    candidate.max_x = max_x;
    candidate.max_y = max_y;
    candidate.indices = std::move(component_indices);

    if (candidate.area > best_any.area) {
      best_any = candidate;
    }
    if (!candidate.touches_border && candidate.area > best_non_border.area) {
      best_non_border = candidate;
    }
  }

  if (best_non_border.found) {
    return best_non_border;
  }
  return best_any;
}

std::vector<float> ResizeBilinear(const std::vector<float>& input, int input_width, int input_height, int output_width, int output_height) {
  std::vector<float> output(static_cast<std::size_t>(output_width * output_height), 0.0F);

  if (input_width <= 0 || input_height <= 0 || output_width <= 0 || output_height <= 0) {
    return output;
  }

  const float scale_x = (output_width > 1) ? static_cast<float>(input_width - 1) / static_cast<float>(output_width - 1)
                                            : 0.0F;
  const float scale_y = (output_height > 1) ? static_cast<float>(input_height - 1) / static_cast<float>(output_height - 1)
                                             : 0.0F;

  for (int y = 0; y < output_height; ++y) {
    const float src_y = scale_y * static_cast<float>(y);
    const int y0 = static_cast<int>(std::floor(src_y));
    const int y1 = std::min(y0 + 1, input_height - 1);
    const float wy = src_y - static_cast<float>(y0);

    for (int x = 0; x < output_width; ++x) {
      const float src_x = scale_x * static_cast<float>(x);
      const int x0 = static_cast<int>(std::floor(src_x));
      const int x1 = std::min(x0 + 1, input_width - 1);
      const float wx = src_x - static_cast<float>(x0);

      const float top_left = input[static_cast<std::size_t>(y0 * input_width + x0)];
      const float top_right = input[static_cast<std::size_t>(y0 * input_width + x1)];
      const float bottom_left = input[static_cast<std::size_t>(y1 * input_width + x0)];
      const float bottom_right = input[static_cast<std::size_t>(y1 * input_width + x1)];

      const float top = top_left + wx * (top_right - top_left);
      const float bottom = bottom_left + wx * (bottom_right - bottom_left);
      output[static_cast<std::size_t>(y * output_width + x)] = top + wy * (bottom - top);
    }
  }

  return output;
}

std::vector<float> ShiftToCenterOfMass(const std::vector<float>& input, int width, int height) {
  std::vector<float> output(static_cast<std::size_t>(width * height), 0.0F);

  double mass = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float value = input[static_cast<std::size_t>(y * width + x)];
      mass += value;
      sum_x += static_cast<double>(x) * value;
      sum_y += static_cast<double>(y) * value;
    }
  }

  if (mass <= std::numeric_limits<double>::epsilon()) {
    return input;
  }

  const double center_x = sum_x / mass;
  const double center_y = sum_y / mass;
  const int shift_x = static_cast<int>(std::round((static_cast<double>(width) - 1.0) / 2.0 - center_x));
  const int shift_y = static_cast<int>(std::round((static_cast<double>(height) - 1.0) / 2.0 - center_y));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int source_x = x - shift_x;
      const int source_y = y - shift_y;
      if (source_x < 0 || source_y < 0 || source_x >= width || source_y >= height) {
        continue;
      }
      output[static_cast<std::size_t>(y * width + x)] = input[static_cast<std::size_t>(source_y * width + source_x)];
    }
  }

  return output;
}

}  // namespace

PreprocessResult PreprocessForMnist(const BmpImage& image) {
  PreprocessResult result;
  result.mnist_input.assign(28U * 28U, 0.0F);

  if (image.width <= 0 || image.height <= 0 || image.rgb.empty()) {
    result.error_message = "Input image is empty.";
    return result;
  }

  const auto grayscale = GaussianBlur3x3(ToGrayscale(image), image.width, image.height);
  const float threshold = ComputeOtsuThreshold(grayscale);
  result.threshold = threshold;

  int dark_pixels = 0;
  int light_pixels = 0;
  const auto dark_mask = MakeForegroundMask(grayscale, threshold, true, &dark_pixels);
  const auto light_mask = MakeForegroundMask(grayscale, threshold, false, &light_pixels);

  const int total_pixels = image.width * image.height;
  const float dark_ratio = static_cast<float>(dark_pixels) / static_cast<float>(total_pixels);
  const float light_ratio = static_cast<float>(light_pixels) / static_cast<float>(total_pixels);

  bool use_dark_foreground = true;
  if (light_ratio < dark_ratio && light_ratio < 0.45F) {
    use_dark_foreground = false;
  }
  result.used_dark_foreground = use_dark_foreground;

  const auto& selected_mask = use_dark_foreground ? dark_mask : light_mask;
  ComponentSelection component = SelectLargestComponent(selected_mask, image.width, image.height);

  if (!component.found) {
    use_dark_foreground = !use_dark_foreground;
    result.used_dark_foreground = use_dark_foreground;
    const auto& fallback_mask = use_dark_foreground ? dark_mask : light_mask;
    component = SelectLargestComponent(fallback_mask, image.width, image.height);
  }

  if (!component.found) {
    result.error_message = "Could not isolate a digit from the captured image. Try a darker marker and a simpler background.";
    return result;
  }

  result.component_area = component.area;
  result.bbox_x = component.min_x;
  result.bbox_y = component.min_y;
  result.bbox_width = component.max_x - component.min_x + 1;
  result.bbox_height = component.max_y - component.min_y + 1;

  const int side = std::max(result.bbox_width, result.bbox_height);
  const int margin = std::max(2, side / 8);
  const int padded_side = side + margin * 2;
  std::vector<float> square(static_cast<std::size_t>(padded_side * padded_side), 0.0F);

  const int bbox_center_x = (component.min_x + component.max_x) / 2;
  const int bbox_center_y = (component.min_y + component.max_y) / 2;
  const int crop_origin_x = bbox_center_x - padded_side / 2;
  const int crop_origin_y = bbox_center_y - padded_side / 2;

  std::vector<std::uint8_t> component_mask(grayscale.size(), 0U);
  for (int index : component.indices) {
    component_mask[static_cast<std::size_t>(index)] = 1U;
  }

  for (int y = 0; y < padded_side; ++y) {
    for (int x = 0; x < padded_side; ++x) {
      const int source_x = crop_origin_x + x;
      const int source_y = crop_origin_y + y;
      if (source_x < 0 || source_y < 0 || source_x >= image.width || source_y >= image.height) {
        continue;
      }
      const std::size_t source_index = static_cast<std::size_t>(source_y * image.width + source_x);
      if (component_mask[source_index] == 0U) {
        continue;
      }
      const float intensity = use_dark_foreground ? (1.0F - grayscale[source_index]) : grayscale[source_index];
      square[static_cast<std::size_t>(y * padded_side + x)] = std::clamp(intensity, 0.0F, 1.0F);
    }
  }

  auto resized = ResizeBilinear(square, padded_side, padded_side, 20, 20);
  std::vector<float> canvas(28U * 28U, 0.0F);
  for (int y = 0; y < 20; ++y) {
    for (int x = 0; x < 20; ++x) {
      canvas[static_cast<std::size_t>((y + 4) * 28 + (x + 4))] = resized[static_cast<std::size_t>(y * 20 + x)];
    }
  }

  canvas = ShiftToCenterOfMass(canvas, 28, 28);

  float max_value = 0.0F;
  for (float value : canvas) {
    max_value = std::max(max_value, value);
  }
  if (max_value > 0.0F) {
    for (float& value : canvas) {
      value = std::clamp(value / max_value, 0.0F, 1.0F);
    }
  }

  result.mnist_input = std::move(canvas);
  result.success = true;
  return result;
}
