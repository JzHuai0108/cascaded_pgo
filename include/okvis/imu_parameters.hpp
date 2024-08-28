#pragma once
#include <Eigen/Core>

namespace okvis {
/*!
 * \brief IMU parameters.
 *
 * A simple struct to specify properties of an IMU.
 *
 */
struct ImuParameters{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double a_max;  ///< Accelerometer saturation. [m/s^2]
  double g_max;  ///< Gyroscope saturation. [rad/s]
  double sigma_g_c;  ///< Gyroscope noise density.
  double sigma_a_c;  ///< Accelerometer noise density.
  double sigma_bg;  ///< Initial gyroscope bias.
  double sigma_ba;  ///< Initial accelerometer bias.
  double sigma_gw_c; ///< Gyroscope drift noise density.
  double sigma_aw_c; ///< Accelerometer drift noise density.
  double tau;  ///< Reversion time constant of accerometer bias. [s]
  double g;  ///< Earth acceleration.
  int rate;  ///< IMU rate in Hz.

  double sigma_Mg_element;  /// sigma for every element in the gyro correction matrix M_g.
  double sigma_Ts_element;
  double sigma_Ma_element;
  // In contrast to gravity direction and camera parameters, whether an IMU
  // parameter is included in the state vector (as a state variable) or not
  // depends on the IMU model (model_name) and does not depend on the sigma of
  // the parameter. This choice I think simplifies the IMU covariance propagation.

  int imuIdx;
  std::string model_name;
  double sigma_gravity_direction; // The uncertainty in both roll and pitch of the gravity direction.

  ImuParameters();

  const Eigen::Vector3d &gravityDirection() const;

  Eigen::Vector3d gravity() const;

  bool isGravityDirectionFixed() const { return sigma_gravity_direction == 0.0; }

  bool isGravityDirectionVariable() const { return sigma_gravity_direction > 0.0; }

  const Eigen::Vector3d &initialGyroBias() const { return g0; }

  const Eigen::Vector3d &initialAccelBias() const { return a0; }

  const Eigen::Matrix<double, 9, 1> &gyroCorrectionMatrix() const { return Mg0; }

  const Eigen::Matrix<double, 9, 1> &gyroGSensitivity() const { return Ts0; }

  const Eigen::Matrix<double, 6, 1> &accelCorrectionMatrix() const { return Ma0; }

  void setGravityDirection(const Eigen::Vector3d &gravityDirection);

  void setInitialGyroBias(const Eigen::Vector3d &gb) { g0 = gb; }

  void setInitialAccelBias(const Eigen::Vector3d &ab) { a0 = ab; }

  void setGyroCorrectionMatrix(const Eigen::Matrix<double, 9, 1> &Mg) {
    Mg0 = Mg;
  }

  void setGyroGSensitivity(const Eigen::Matrix<double, 9, 1> &Ts) { Ts0 = Ts; }

  void setAccelCorrectionMatrix(const Eigen::Matrix<double, 6, 1> &Ma) {
    Ma0 = Ma;
  }

  std::string toString(const std::string &hint) const;

private:
  /// prior knowledge of IMU intrinsic parameters.
  Eigen::Vector3d g0;  ///< Mean of the prior gyroscope bias.
  Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
  Eigen::Matrix<double, 9, 1> Mg0;
  Eigen::Matrix<double, 9, 1> Ts0;
  Eigen::Matrix<double, 6, 1> Ma0;

  Eigen::Vector3d normalGravity;
};

typedef ImuParameters ImuNoiseParameters;

ImuParameters createX36DImuParameters();

} // namespace okvis