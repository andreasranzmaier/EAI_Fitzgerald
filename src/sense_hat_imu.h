#pragma once

#include <cstdint>
#include <memory>
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

// Reads the Sense HAT LSM9DS1 via RTIMULib.
// Uses /etc/RTIMULib.ini (installed by the sense-hat package) so axis
// orientation and calibration match the Python library used for training.
class SenseHatImu {
 public:
  SenseHatImu();
  ~SenseHatImu();

  bool available() const { return available_; }
  const std::string& error_message() const { return error_message_; }

  // Fills *sample with the current sensor state. Returns false on I/O error.
  bool Read(ImuSample* sample);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool available_ = false;
  std::string error_message_;
};
