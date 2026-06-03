#include "sense_hat_imu.h"

#include <chrono>
#include <fstream>
#include <string>

namespace {

// IIO device names accepted as the Sense HAT IMU.
constexpr const char* kKnownNames[] = {"lsm9ds1_imu", "lsm6ds0_imu", nullptr};

// IIO accel scale gives m/s² per LSB; training data uses g-force.
constexpr float kGravity = 9.80665f;

// IIO LSM9DS1 mag scale gives Gauss per LSB; SenseHat Python library returns µT.
// 1 Gauss = 100 µT.  Set to 1.0 if the kernel driver already reports µT.
constexpr float kGaussToMicrotesla = 100.0f;

std::string TrimTrailing(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) //
    s.pop_back();
  return s;
}

}  // namespace

SenseHatImu::SenseHatImu() {
  available_ = FindDevice();
}

bool SenseHatImu::FindDevice() {
  for (int i = 0; i < 16; ++i) {
    const std::string base = "/sys/bus/iio/devices/iio:device" + std::to_string(i);
    std::ifstream name_file(base + "/name");
    if (!name_file) continue;

    std::string name;
    std::getline(name_file, name);
    name = TrimTrailing(name);

    bool matched = false;
    for (int j = 0; kKnownNames[j] != nullptr; ++j) {
      if (name == kKnownNames[j]) {
        matched = true;
        break;
      }
    }
    if (!matched) continue;

    device_path_ = base;

    accel_scale_ = ReadFloat(base + "/in_accel_scale");
    gyro_scale_  = ReadFloat(base + "/in_anglvel_scale");
    mag_scale_   = ReadFloat(base + "/in_magn_scale");

    // Fall back to 1.0 so raw values are still returned if scale is missing.
    if (accel_scale_ == 0.0f) accel_scale_ = 1.0f;
    if (gyro_scale_  == 0.0f) gyro_scale_  = 1.0f;
    if (mag_scale_   == 0.0f) mag_scale_   = 1.0f;

    return true;
  }
  error_message_ = "Sense HAT IMU IIO device not found (tried lsm9ds1_imu, lsm6ds0_imu).";
  return false;
}

float SenseHatImu::ReadFloat(const std::string& path) const {
  std::ifstream file(path);
  if (!file) return 0.0f;
  float value = 0.0f;
  file >> value;
  return value;
}

float SenseHatImu::ReadRawScaled(const std::string& raw_path, float scale) const {
  std::ifstream file(raw_path);
  if (!file) return 0.0f;
  int raw = 0;
  file >> raw;
  return static_cast<float>(raw) * scale;
}

bool SenseHatImu::Read(ImuSample* sample) {
  using Clock = std::chrono::steady_clock;
  using Ms    = std::chrono::milliseconds;
  sample->timestamp_ms = std::chrono::duration_cast<Ms>(Clock::now().time_since_epoch()).count();

  // raw × scale (m/s²) ÷ gravity -> g-force
  sample->accel_x = ReadRawScaled(device_path_ + "/in_accel_x_raw", accel_scale_) / kGravity;
  sample->accel_y = ReadRawScaled(device_path_ + "/in_accel_y_raw", accel_scale_) / kGravity;
  sample->accel_z = ReadRawScaled(device_path_ + "/in_accel_z_raw", accel_scale_) / kGravity;

  // raw × scale -> rad/s (matches training data directly)
  sample->gyro_x = ReadRawScaled(device_path_ + "/in_anglvel_x_raw", gyro_scale_);
  sample->gyro_y = ReadRawScaled(device_path_ + "/in_anglvel_y_raw", gyro_scale_);
  sample->gyro_z = ReadRawScaled(device_path_ + "/in_anglvel_z_raw", gyro_scale_);

  // raw × scale (Gauss) × 100 -> µT
  sample->mag_x = ReadRawScaled(device_path_ + "/in_magn_x_raw", mag_scale_) * kGaussToMicrotesla;
  sample->mag_y = ReadRawScaled(device_path_ + "/in_magn_y_raw", mag_scale_) * kGaussToMicrotesla;
  sample->mag_z = ReadRawScaled(device_path_ + "/in_magn_z_raw", mag_scale_) * kGaussToMicrotesla;

  return true;
}
