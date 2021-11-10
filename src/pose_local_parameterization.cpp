#include "pose_local_parameterization.h"
#include <Eigen/Geometry>

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar>
deltaQ(const Eigen::MatrixBase<Derived> &theta) {
  typedef typename Derived::Scalar Scalar_t;

  Eigen::Quaternion<Scalar_t> dq;
  Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
  half_theta /= static_cast<Scalar_t>(2.0);
  dq.w() = static_cast<Scalar_t>(1.0);
  dq.x() = half_theta.x();
  dq.y() = half_theta.y();
  dq.z() = half_theta.z();
  return dq;
}

bool PoseLocalParameterization::Plus(const double *x, const double *delta,
                                     double *x_plus_delta) const {
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

  Eigen::Map<const Eigen::Vector3d> dp(delta);
  Eigen::Quaterniond dq(deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3)));

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;
  q = dq * _q;

  return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x,
                                                double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
  j.setIdentity();
  return true;
}
