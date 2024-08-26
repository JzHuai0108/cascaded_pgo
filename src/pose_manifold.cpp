#include "pose_manifold.h"

namespace ceres {

// Generalization of the addition operation,
//        x_plus_delta = Plus(x, delta)
//        with the condition that Plus(x, 0) = x.
bool PoseManifold::Plus(const double* x, const double* delta,
                                     double* x_plus_delta) const {
  return plus(x, delta, x_plus_delta);
}

// Generalization of the addition operation,
//        x_plus_delta = Plus(x, delta)
//        with the condition that Plus(x, 0) = x.
bool PoseManifold::plus(const double* x, const double* delta,
                                     double* x_plus_delta) {

  Eigen::Map<const Eigen::Matrix<double, 6, 1> > delta_(delta);
  Eigen::Vector3d r0(x[0], x[1], x[2]);
  Eigen::Quaterniond q0(x[6], x[3], x[4], x[5]);

  Eigen::Vector3d r = r0 + delta_.head<3>();
  x_plus_delta[0] = r[0];
  x_plus_delta[1] = r[1];
  x_plus_delta[2] = r[2];
  Eigen::Matrix<double, 3, 1> dq = delta_.tail<3>();
  Eigen::Quaterniond q = sophus::expAndTheta(dq) * q0;
  x_plus_delta[3] = q.x();
  x_plus_delta[4] = q.y();
  x_plus_delta[5] = q.z();
  x_plus_delta[6] = q.w();

  return true;
}

// Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
bool PoseManifold::Minus(const double* x_plus_delta, const double* x, double* delta) const {
  return minus(x_plus_delta, x, delta);
}

// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool PoseManifold::ComputeLiftJacobian(const double* x,
                                                    double* jacobian) const {
  return liftJacobian(x, jacobian);
}

// Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
bool PoseManifold::minus(const double* x_plus_delta, const double* x, double* delta) {
  delta[0] = x_plus_delta[0] - x[0];
  delta[1] = x_plus_delta[1] - x[1];
  delta[2] = x_plus_delta[2] - x[2];
  const Eigen::Quaterniond q_plus_delta_(x_plus_delta[6], x_plus_delta[3],
                                         x_plus_delta[4], x_plus_delta[5]);
  const Eigen::Quaterniond q_(x[6], x[3], x[4], x[5]);
  Eigen::Map<Eigen::Vector3d> delta_q_(&delta[3]);
  double theta;

  // delta_q_ = 2 * (q_plus_delta_ * q_.inverse()).coeffs().template head<3>();
  delta_q_ = sophus::logAndTheta(q_plus_delta_ * q_.inverse(), &theta);
  return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
bool PoseManifold::plusJacobian(const double* x,
                                             double* jacobian) {
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
  Eigen::Vector3d r0(x[0], x[1], x[2]);
  Eigen::Quaterniond q0(x[6], x[3], x[4], x[5]);

  Eigen::Matrix<double, 4, 3> S = Eigen::Matrix<double, 4, 3>::Zero();
  Jp.setZero();
  Jp.topLeftCorner<3, 3>().setIdentity();
  S(0, 0) = 0.5;
  S(1, 1) = 0.5;
  S(2, 2) = 0.5;
  Jp.bottomRightCorner<4, 3>() = oplusQuat(q0) * S;
  return true;
}

// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool PoseManifold::liftJacobian(const double* x,
                                             double* jacobian) {

  Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_lift(jacobian);
  const Eigen::Quaterniond q_inv(x[6], -x[3], -x[4], -x[5]);
  J_lift.setZero();
  J_lift.topLeftCorner<3, 3>().setIdentity();
  Eigen::Matrix4d Qplus = oplusQuat(q_inv);
  Eigen::Matrix<double, 3, 4> Jq_pinv;
  Jq_pinv.bottomRightCorner<3, 1>().setZero();
  Jq_pinv.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 2.0;
  J_lift.bottomRightCorner<3, 4>() = Jq_pinv * Qplus;

  return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
bool PoseManifold::ComputeJacobian(const double* x,
                                                double* jacobian) const {

  return plusJacobian(x, jacobian);
}

bool PoseManifold::VerifyJacobianNumDiff(const double* x,
                                                      double* jacobian,
                                                      double* jacobianNumDiff) {
  plusJacobian(x, jacobian);
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jpn(
      jacobianNumDiff);
  double dx = 1e-9;
  Eigen::Matrix<double, 7, 1> xp;
  Eigen::Matrix<double, 7, 1> xm;
  for (size_t i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> delta;
    delta.setZero();
    delta[i] = dx;
    Plus(x, delta.data(), xp.data());
    delta[i] = -dx;
    Plus(x, delta.data(), xm.data());
    Jpn.col(i) = (xp - xm) / (2 * dx);
  }
  if ((Jp - Jpn).norm() < 1e-6)
    return true;
  else
    return false;
}

} // namespace ceres