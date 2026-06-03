#include "sense_hat_imu.h"

#include <chrono>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace {

constexpr const char* kI2cDev  = "/dev/i2c-1";
constexpr std::uint8_t kAddrAG  = 0x6A;  // LSM9DS1 accel + gyro
constexpr std::uint8_t kAddrMag = 0x1C;  // LSM9DS1 magnetometer

// AG registers
constexpr std::uint8_t kRegCtrl1G    = 0x10;  // gyro ODR / FS
constexpr std::uint8_t kRegCtrl6XL   = 0x20;  // accel ODR / FS
constexpr std::uint8_t kRegOutGyroX  = 0x18;  // gyro  X low byte (6 bytes)
constexpr std::uint8_t kRegOutAccX   = 0x28;  // accel X low byte (6 bytes)

// Mag registers
constexpr std::uint8_t kRegCtrl1M  = 0x20;
constexpr std::uint8_t kRegCtrl2M  = 0x21;
constexpr std::uint8_t kRegCtrl3M  = 0x22;
constexpr std::uint8_t kRegOutMagX = 0x28;  // mag X low byte (6 bytes)

// Scale factors matching RTIMULib LSM9DS1 defaults used by sense-hat Python library:
//   Gyro  ±500 dps → 17.5 mdps/LSB → rad/s/LSB
constexpr float kGyroScale  = 0.0175f / 180.0f * 3.14159265f;
//   Accel ±8g      → 0.244 mg/LSB  → g/LSB
constexpr float kAccelScale = 0.000244f;
//   Mag   ±4 Gauss → 0.14 mG/LSB × 100 µT/G → µT/LSB
constexpr float kMagScale   = 0.014f;

int16_t ToInt16(std::uint8_t lo, std::uint8_t hi) {
  return static_cast<int16_t>((static_cast<std::uint16_t>(hi) << 8) | lo);
}

}  // namespace

SenseHatImu::SenseHatImu() {
  available_ = InitDevice();
}

SenseHatImu::~SenseHatImu() {
  if (i2c_fd_ >= 0) close(i2c_fd_);
}

bool SenseHatImu::WriteReg(std::uint8_t addr, std::uint8_t reg, std::uint8_t val) {
  if (ioctl(i2c_fd_, I2C_SLAVE, addr) < 0) return false;
  const std::uint8_t buf[2] = {reg, val};
  return write(i2c_fd_, buf, 2) == 2;
}

bool SenseHatImu::ReadRegs(std::uint8_t addr, std::uint8_t reg, std::uint8_t* buf, int len) {
  if (ioctl(i2c_fd_, I2C_SLAVE, addr) < 0) return false;
  if (write(i2c_fd_, &reg, 1) != 1) return false;
  return read(i2c_fd_, buf, len) == len;
}

bool SenseHatImu::InitDevice() {
  i2c_fd_ = open(kI2cDev, O_RDWR);
  if (i2c_fd_ < 0) {
    error_message_ = "Cannot open " + std::string(kI2cDev);
    return false;
  }

  // Gyro: ODR=119Hz, FS=±500dps  CTRL_REG1_G [7:5]=011 [4:3]=01 → 0x68
  if (!WriteReg(kAddrAG, kRegCtrl1G, 0x68)) {
    error_message_ = "Failed to configure gyroscope";
    return false;
  }
  // Accel: ODR=119Hz, FS=±8g     CTRL_REG6_XL [7:5]=011 [4:3]=11 → 0x6C
  if (!WriteReg(kAddrAG, kRegCtrl6XL, 0x6C)) {
    error_message_ = "Failed to configure accelerometer";
    return false;
  }
  // Mag: ODR=10Hz                 CTRL_REG1_M [4:2]=100 → 0x10
  if (!WriteReg(kAddrMag, kRegCtrl1M, 0x10)) {
    error_message_ = "Failed to configure magnetometer (ctrl1)";
    return false;
  }
  // Mag: FS=±4 Gauss              CTRL_REG2_M [6:5]=00 → 0x00
  if (!WriteReg(kAddrMag, kRegCtrl2M, 0x00)) {
    error_message_ = "Failed to configure magnetometer (ctrl2)";
    return false;
  }
  // Mag: continuous-conversion    CTRL_REG3_M [1:0]=00 → 0x00
  if (!WriteReg(kAddrMag, kRegCtrl3M, 0x00)) {
    error_message_ = "Failed to configure magnetometer (ctrl3)";
    return false;
  }

  return true;
}

bool SenseHatImu::Read(ImuSample* sample) {
  using Clock = std::chrono::steady_clock;
  using Ms    = std::chrono::milliseconds;
  sample->timestamp_ms = std::chrono::duration_cast<Ms>(Clock::now().time_since_epoch()).count();

  std::uint8_t buf[6];

  if (!ReadRegs(kAddrAG, kRegOutGyroX, buf, 6)) return false;
  sample->gyro_x = ToInt16(buf[0], buf[1]) * kGyroScale;
  sample->gyro_y = ToInt16(buf[2], buf[3]) * kGyroScale;
  sample->gyro_z = ToInt16(buf[4], buf[5]) * kGyroScale;

  if (!ReadRegs(kAddrAG, kRegOutAccX, buf, 6)) return false;
  sample->accel_x = ToInt16(buf[0], buf[1]) * kAccelScale;
  sample->accel_y = ToInt16(buf[2], buf[3]) * kAccelScale;
  sample->accel_z = ToInt16(buf[4], buf[5]) * kAccelScale;

  if (!ReadRegs(kAddrMag, kRegOutMagX, buf, 6)) return false;
  sample->mag_x = ToInt16(buf[0], buf[1]) * kMagScale;
  sample->mag_y = ToInt16(buf[2], buf[3]) * kMagScale;
  sample->mag_z = ToInt16(buf[4], buf[5]) * kMagScale;

  return true;
}
