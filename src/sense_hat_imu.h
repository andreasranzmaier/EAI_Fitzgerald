
#pragma once

#include <cstdint>
#include <string>

// One IMU reading from the Sense HAT LSM9DS1.
// Units match the SenseHat Python library (and the training CSV recordings):
//   accel  – g-force
//   gyro   – rad/s
//   mag    – µT (micro-Tesla)
struct ImuSample {
  int64_t timestamp_ms;
  float accel_x, accel_y, accel_z;
  float gyro_x,  gyro_y,  gyro_z;
  float mag_x,   mag_y,   mag_z;
};

// Reads the Sense HAT IMU via the Linux IIO sysfs interface.
// Searches for a known device name under /sys/bus/iio/devices/iio:deviceN/.
class SenseHatImu {
 public:
  SenseHatImu();

  bool available() const { return available_; }
  const std::string& error_message() const { return error_message_; }

  // Fills *sample with the current sensor state.  Returns false on I/O error.
  bool Read(ImuSample* sample);

 private:
  bool FindDevice();
  float ReadFloat(const std::string& path) const;
  float ReadRawScaled(const std::string& raw_path, float scale) const;

  std::string device_path_;
  float accel_scale_ = 1.0f;  // IIO raw -> m/s²
  float gyro_scale_  = 1.0f;  // IIO raw -> rad/s
  float mag_scale_   = 1.0f;  // IIO raw -> Gauss (converted to µT on Read)

  bool available_ = false;
  std::string error_message_;
};
