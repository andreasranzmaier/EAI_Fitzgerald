
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

// Reads the Sense HAT LSM9DS1 directly via /dev/i2c-1.
// AG at 0x6A, magnetometer at 0x1C — confirmed by i2cdetect on the target Pi.
class SenseHatImu {
 public:
  SenseHatImu();
  ~SenseHatImu();

  bool available() const { return available_; }
  const std::string& error_message() const { return error_message_; }

  // Fills *sample with the current sensor state.  Returns false on I/O error.
  bool Read(ImuSample* sample);

 private:
  bool InitDevice();
  bool WriteReg(std::uint8_t addr, std::uint8_t reg, std::uint8_t val);
  bool ReadRegs(std::uint8_t addr, std::uint8_t reg, std::uint8_t* buf, int len);

  int i2c_fd_ = -1;
  bool available_ = false;
  std::string error_message_;
};
