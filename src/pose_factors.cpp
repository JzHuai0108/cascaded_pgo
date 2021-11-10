#include "pose_factors.h"
#include "hamilton_quaternion.h"

bool EdgeSE3::Evaluate(double const *const *parameters, double *residuals,
                       double **jacobians) const {
  Eigen::Map<const Eigen::Vector3d> p1(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> q1(parameters[0] + 3);

  Eigen::Map<const Eigen::Vector3d> p2(parameters[1]);
  Eigen::Map<const Eigen::Quaterniond> q2(parameters[1] + 3);

  Eigen::Vector3d p12 = q1.inverse() * (p2 - p1);
  Eigen::Quaterniond q12 = q1.inverse() * q2;

  Eigen::Map<Eigen::Matrix<double, 6, 1>> error(residuals);

  error.head<3>() = p12 - p_ab_;
  Eigen::Quaterniond dq = q12 * q_ab_.inverse();
  Eigen::AngleAxisd aa(dq);
  error.tail<3>() = aa.axis() * aa.angle();

  if (jacobians != NULL) {
    Eigen::Matrix3d invR1 = q1.inverse().toRotationMatrix();
    Eigen::Vector3d eRot = error.tail<3>();
    Eigen::Matrix3d invLeftJacobian = hamilton::logDiffMat<double>(eRot);
    Eigen::Matrix3d product = invLeftJacobian * invR1;
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0.setZero();
      J0.block<3, 3>(0, 0) = -invR1;
      Eigen::Vector3d dp = p2 - p1;
      J0.block<3, 3>(0, 3) = invR1 * hamilton::crossMx(dp);
      J0.block<3, 3>(3, 3) = -product;
    }
    if (jacobians[1] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J1(jacobians[1]);
      J1.setZero();
      J1.block<3, 3>(0, 0) = invR1;
      J1.block<3, 3>(3, 3) = product;
    }
  }
  return true;
}

bool EdgeSE3Prior::Evaluate(double const *const *parameters, double *residuals,
                            double **jacobians) const {

  Eigen::Map<const Eigen::Vector3d> p(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> q(parameters[0] + 3);

  Eigen::Map<Eigen::Matrix<double, 6, 1>> error(residuals);
  error.head<3>() = p - p_;
  Eigen::Quaterniond dq = q * q_.inverse();
  Eigen::AngleAxisd aa(dq);
  error.tail<3>() = aa.axis() * aa.angle();

  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0.setIdentity();
      Eigen::Vector3d eRot = error.tail<3>();
      J0.block<3, 3>(3, 3) = hamilton::logDiffMat<double>(eRot);
    }
  }
  return true;
}
