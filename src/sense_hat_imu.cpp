#include "sense_hat_imu.h"

#include <chrono>
#include <thread>

#include <RTIMULib.h>

struct SenseHatImu::Impl {
  RTIMUSettings* settings = nullptr;
  RTIMU* imu = nullptr;

  ~Impl() {
    delete imu;
    delete settings;
  }
};

SenseHatImu::SenseHatImu() : impl_(std::make_unique<Impl>()) {
  // /etc/RTIMULib.ini is installed by the sense-hat package and carries the
  // same axis rotation and calibration the Python library used for recording.
  impl_->settings = new RTIMUSettings("/etc", "RTIMULib");
  impl_->imu = RTIMU::createIMU(impl_->settings);

  if (!impl_->imu || impl_->imu->IMUType() == RTIMU_TYPE_NULL) {
    error_message_ = "No IMU detected - check I2C bus and /etc/RTIMULib.ini";
    return;
  }
  if (!impl_->imu->IMUInit()) {
    error_message_ = "IMU init failed";
    return;
  }
  available_ = true;
}

SenseHatImu::~SenseHatImu() = default;

bool SenseHatImu::Read(ImuSample* sample) {
  using Clock = std::chrono::steady_clock;
  using Ms    = std::chrono::milliseconds;

  // IMURead() returns false when the sensor hasn't produced a new sample yet.
  // Retry briefly to handle calls slightly ahead of the sensor ODR (~119 Hz).
  for (int retry = 0; retry < 5; ++retry) {
    if (impl_->imu->IMURead()) {
      const RTIMU_DATA& data = impl_->imu->getIMUData();
      sample->timestamp_ms = std::chrono::duration_cast<Ms>(
          Clock::now().time_since_epoch()).count();
      sample->accel_x = static_cast<float>(data.accel.x());
      sample->accel_y = static_cast<float>(data.accel.y());
      sample->accel_z = static_cast<float>(data.accel.z());
      sample->gyro_x  = static_cast<float>(data.gyro.x());
      sample->gyro_y  = static_cast<float>(data.gyro.y());
      sample->gyro_z  = static_cast<float>(data.gyro.z());
      sample->mag_x   = static_cast<float>(data.compass.x());
      sample->mag_y   = static_cast<float>(data.compass.y());
      sample->mag_z   = static_cast<float>(data.compass.z());
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  error_message_ = "IMU read timeout";
  return false;
}
