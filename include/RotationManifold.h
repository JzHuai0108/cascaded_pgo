#include <ceres/manifold.h>

#include <bsplines_ceres/pose_manifold.h>

namespace bsplines {
namespace ceres {
/**
 * The manifold used by ceres solver for SO3.
 * This implementation assumes right invariant error, i.e., \f$ R_{WB} = exp(\epsilon) \hat{R}_{WB} \f$.
 * This error agrees with the error convention in computing analytic Jacobians in SO3BSplines.
 * This implementation computes a dummy identity but working PlusJacobian to satisfy ceres solver.
 * This is due to the fact that the analytic Jacobians by SO3BSplines are relative to the minimal space perturbation.
 * @warning As a result, if you use the AutoDiffCostFunction which computes Jacobians relative to the ambient space perturbation,
 * be sure to use RotationManifold.
*/
class RotationManifoldSimplified : public ::ceres::Manifold {
public:
  ~RotationManifoldSimplified() {}
  bool Plus(const double *x, const double *delta,
            double *x_plus_delta) const final {
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_(delta);
    Eigen::Vector4d R(x[0], x[1], x[2], x[3]);
    Eigen::Map<Eigen::Matrix<double, 4, 1>> Rplus(x_plus_delta);
    Rplus = hamilton::qplus(hamilton::axisAngle2quat(delta_), R);

    return true;
  }
  bool PlusJacobian(const double * /*x*/, double *jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> Jp(jacobian);
    Jp.setIdentity();
    return true;
  }

  bool RightMultiplyByPlusJacobian(const double * /*x*/, const int num_rows,
                                   const double *ambient_matrix,
                                   double *tangent_matrix) const final {
    Eigen::Map<const Eigen::Matrix<double, -1, 4, Eigen::RowMajor>> am(ambient_matrix, num_rows, 4);
    Eigen::Map<Eigen::Matrix<double, -1, 3, Eigen::RowMajor>> tm(tangent_matrix, num_rows, 3);
    tm = am.leftCols<3>();
    return true;
  }

  bool Minus(const double* y, const double* x, double* y_minus_x) const final {
    Eigen::Map<const Eigen::Matrix<double, 4, 1>> ym(y);
    Eigen::Map<const Eigen::Matrix<double, 4, 1>> xm(x);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> delta(y_minus_x);
    delta = hamilton::quat2AxisAngle(hamilton::qplus(ym, hamilton::quatInv(xm)));
    return true;
  }

  bool MinusJacobian(const double* /*x*/, double* jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Jp(jacobian);
    Jp.setIdentity();
    return true;
  }

  int AmbientSize() const { return 4; }

  int TangentSize() const { return 3; }
};

class RotationManifold : public ::ceres::Manifold {
public:
  ~RotationManifold() {}
  bool Plus(const double *x, const double *delta,
            double *x_plus_delta) const final {
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_(delta);
    Eigen::Vector4d R(x[0], x[1], x[2], x[3]);
    Eigen::Map<Eigen::Matrix<double, 4, 1>> Rplus(x_plus_delta);
    Rplus = hamilton::qplus(hamilton::axisAngle2quat(delta_), R);
    return true;
  }

  bool PlusJacobian(const double * x, double *jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> Jp(jacobian);
    Eigen::Matrix<double, 4, 3> S = Eigen::Matrix<double, 4, 3>::Zero();
    S(0, 0) = 0.5;
    S(1, 1) = 0.5;
    S(2, 2) = 0.5;
    Eigen::Map<const Eigen::Quaterniond> q0(x);
    Jp = oplusQuat(q0) * S;
    return true;
  }

  bool Minus(const double* y, const double* x, double* y_minus_x) const final {
    Eigen::Map<const Eigen::Matrix<double, 4, 1>> ym(y);
    Eigen::Map<const Eigen::Matrix<double, 4, 1>> xm(x);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> delta(y_minus_x);
    delta = hamilton::quat2AxisAngle(hamilton::qplus(ym, hamilton::quatInv(xm)));
    return true;
  }

  bool MinusJacobian(const double* x, double* jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Jp(jacobian);
    const Eigen::Quaterniond q_inv(x[3], -x[0], -x[1], -x[2]);
    Eigen::Matrix4d Qplus = oplusQuat(q_inv);
    Eigen::Matrix<double, 3, 4> Jq_pinv;
    Jq_pinv.bottomRightCorner<3, 1>().setZero();
    Jq_pinv.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 2.0;
    Jp = Jq_pinv * Qplus;
    return true;
  }

  int AmbientSize() const { return 4; }

  int TangentSize() const { return 3; }
};
}  // namespace ceres
}  // namespace bsplines