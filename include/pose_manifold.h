#ifndef CERES_POSE_MANIFOLD_H_
#define CERES_POSE_MANIFOLD_H_

#include <ceres/manifold.h>
#include <Eigen/Geometry>

namespace sophus {

/// Warn: Do not use sinc or its templated version for autodiff involving quaternions
///  as its real part may be calculated without considering the infinisimal input.
/// Use expAndTheta borrowed from Sophus instead for this purpose
template <typename Scalar>
Eigen::Quaternion<Scalar> expAndTheta(const Eigen::Matrix<Scalar, 3, 1> & omega) {
    Scalar theta_sq = omega.squaredNorm();
    Scalar theta = sqrt(theta_sq);
    Scalar half_theta = static_cast<Scalar>(0.5)*(theta);

    Scalar imag_factor;
    Scalar real_factor;
    if(theta < static_cast<Scalar>(1e-10)) {
      Scalar theta_po4 = theta_sq*theta_sq;
      imag_factor = static_cast<Scalar>(0.5)
                    - static_cast<Scalar>(1.0/48.0)*theta_sq
                    + static_cast<Scalar>(1.0/3840.0)*theta_po4;
      real_factor = static_cast<Scalar>(1)
                    - static_cast<Scalar>(0.5)*theta_sq +
                    static_cast<Scalar>(1.0/384.0)*theta_po4;
    } else {
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta/theta;
      real_factor = cos(half_theta);
    }

    return Eigen::Quaternion<Scalar>(real_factor,
                                               imag_factor*omega.x(),
                                               imag_factor*omega.y(),
                                               imag_factor*omega.z());
}

// From sophus so3.hpp
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> logAndTheta(const Eigen::Quaternion<Scalar> & other,
                          Scalar * theta) {
  Scalar squared_n
      = other.vec().squaredNorm();
  Scalar n = sqrt(squared_n);
  Scalar w = other.w();

  Scalar two_atan_nbyw_by_n;

  // Atan-based log thanks to
  //
  // C. Hertzberg et al.:
  // "Integrating Generic Sensor Fusion Algorithms with Sound State
  // Representation through Encapsulation of Manifolds"
  // Information Fusion, 2011

  if (n < static_cast<Scalar>(1e-10)) {
    // If quaternion is normalized and n=0, then w should be 1;
    // w=0 should never happen here!
    CHECK_GT(abs(w), static_cast<Scalar>(1e-10)) <<
                  "Quaternion should be normalized!";
    Scalar squared_w = w*w;
    two_atan_nbyw_by_n = static_cast<Scalar>(2) / w
                         - static_cast<Scalar>(2)*(squared_n)/(w*squared_w);
  } else {
    if (abs(w) < static_cast<Scalar>(1e-10)) {
      if (w > static_cast<Scalar>(0)) {
        two_atan_nbyw_by_n = M_PI/n;
      } else {
        two_atan_nbyw_by_n = -M_PI/n;
      }
    }else{
      two_atan_nbyw_by_n = static_cast<Scalar>(2) * atan(n/w) / n;
    }
  }

  *theta = two_atan_nbyw_by_n*n;

  return two_atan_nbyw_by_n * other.vec();
}
} // namespace sophus

namespace ceres {
/// \brief Oplus matrix of a quaternion, i.e. q_AB*q_BC = oplus(q_BC)*q_AB.coeffs().
/// @param[in] q_BC A Quaternion.
inline Eigen::Matrix4d oplusQuat(const Eigen::Quaterniond & q_BC) {
  Eigen::Vector4d q = q_BC.coeffs();
  Eigen::Matrix4d Q;
  Q(0,0) =  q[3]; Q(0,1) =  q[2]; Q(0,2) = -q[1]; Q(0,3) =  q[0];
  Q(1,0) = -q[2]; Q(1,1) =  q[3]; Q(1,2) =  q[0]; Q(1,3) =  q[1];
  Q(2,0) =  q[1]; Q(2,1) = -q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
  return Q;
}

// This is different from PoseManifold in liftJacobian() and
// ComputeJacobian() in that both are set to identity here.
class PoseManifoldSimplified : public ::ceres::Manifold {
public:
  static const size_t kNumParams = 6;
  static const size_t kGlobalDim = 7;
  virtual ~PoseManifoldSimplified() {}

  static inline int getMinimalDim() { return kNumParams; }

  //  static void updateState(const Eigen::Vector3d &r, const Eigen::Quaterniond
  //  &q,
  //                          const Eigen::VectorXd &delta,
  //                          Eigen::Vector3d *r_delta,
  //                          Eigen::Quaterniond *q_delta) {
  //    Eigen::Vector3d deltaAlpha = delta.segment<3>(3);
  //    *r_delta = r + delta.head<3>();
  //    *q_delta = expAndTheta(deltaAlpha) * q;
  //    q_delta->normalize();
  //  }

  template <typename Scalar>
  static void
  oplus(const Scalar *const deltaT,
        std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> *T) {
    Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> delta(deltaT);
    T->first += delta.template head<3>();
    Eigen::Matrix<Scalar, 3, 1> omega = delta.template tail<3>();
    Eigen::Quaternion<Scalar> dq = sophus::expAndTheta(omega);
    T->second = dq * T->second;
  }

  template <typename Scalar>
  static void oplus(const Scalar *x, const Scalar *delta,
                    Scalar *x_plus_delta) {
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> dr(delta);
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> dq(delta + 3);
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> r(x);
    Eigen::Map<const Eigen::Quaternion<Scalar>> q(x + 3);
    Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> rplus(x_plus_delta);
    Eigen::Map<Eigen::Quaternion<Scalar>> qplus(x_plus_delta + 3);
    rplus = r + dr;
    qplus = sophus::expAndTheta(Eigen::Matrix<Scalar, 3, 1>(dq)) * q;
  }

  template <typename Scalar>
  static void ominus(const Scalar *T_plus, const Scalar *T, Scalar *delta) {
    delta[0] = T_plus[0] - T[0];
    delta[1] = T_plus[1] - T[1];
    delta[2] = T_plus[2] - T[2];
    const Eigen::Quaterniond q_plus(T_plus[6], T_plus[3], T_plus[4], T_plus[5]);
    const Eigen::Quaterniond q(T[6], T[3], T[4], T[5]);
    Eigen::Map<Eigen::Vector3d> delta_q(&delta[3]);
    double theta;
    delta_q = sophus::logAndTheta(q_plus * q.inverse(), &theta);
  }

  bool Minus(const double *x_plus_delta, const double *x,
             double *delta) const final {
    ominus(x_plus_delta, x, delta);
    return true;
  }


  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  ///     It must be the pseudo inverse of the local parametrization Jacobian as
  ///     obtained by COmputeJacobian.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double * /*x*/, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, kNumParams, kGlobalDim, Eigen::RowMajor>>
        J_lift(jacobian);
    J_lift.setIdentity();
    return true;
  }

  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  bool MinusJacobian(const double *x, double *jacobian) const {
    return liftJacobian(x, jacobian);
  }

  bool Plus(const double *x, const double *delta,
            double *x_plus_delta) const final {
    // transform to rao framework
    std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> T(
        Eigen::Vector3d(x[0], x[1], x[2]),
        Eigen::Quaterniond(x[6], x[3], x[4], x[5]));
    // call oplus operator in rao
    oplus(delta, &T);

    // copy back
    const Eigen::Vector3d &r = T.first;
    x_plus_delta[0] = r[0];
    x_plus_delta[1] = r[1];
    x_plus_delta[2] = r[2];
    const Eigen::Vector4d &q = T.second.coeffs();
    x_plus_delta[3] = q[0];
    x_plus_delta[4] = q[1];
    x_plus_delta[5] = q[2];
    x_plus_delta[6] = q[3];
    return true;
  }

  bool ComputeJacobian(const double * /*x*/, double *jacobian) const {
    Eigen::Map<Eigen::Matrix<double, kGlobalDim, kNumParams, Eigen::RowMajor>>
        j(jacobian);
    j.setIdentity();
    return true;
  }

  virtual int GlobalSize() const { return kGlobalDim; }
  virtual int LocalSize() const { return kNumParams; }

  int AmbientSize() const final { return kGlobalDim; };
  int TangentSize() const final { return kNumParams; };

  bool PlusJacobian(const double* x, double* jacobian) const final {
    return ComputeJacobian(x, jacobian);
  }
};

/// \brief Pose local parameterisation, i.e. for orientation dq(dalpha) x q_bar.
class PoseManifold : public ::ceres::Manifold {
 public:

  /// \brief Trivial destructor.
  virtual ~PoseManifold() {
  }

  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x Variable.
  /// @param[in] delta Perturbation.
  /// @param[out] x_plus_delta Perturbed x.
  bool Plus(const double* x, const double* delta,
                    double* x_plus_delta) const final;

  /// \brief Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
  /// @param[in] x_plus_delta Perturbed variable.
  /// @param[in] x Variable.
  /// @param[out] delta minimal difference.
  /// \return True on success.
  bool Minus(const double* x_plus_delta, const double* x,
                     double* delta) const final;

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  /// @param[in] x Variable.
  /// @param[out] jacobian The Jacobian.
  bool ComputeJacobian(const double* x, double* jacobian) const;

  /// \brief Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  bool ComputeLiftJacobian(const double* x, double* jacobian) const;

  // provide these as static for easy use elsewhere:

  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x Variable.
  /// @param[in] delta Perturbation.
  /// @param[out] x_plus_delta Perturbed x.
  static bool plus(const double* x, const double* delta, double* x_plus_delta);

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  /// @param[in] x Variable.
  /// @param[out] jacobian The Jacobian.
  static bool plusJacobian(const double* x, double* jacobian);

  /// \brief Computes the minimal difference between a variable x and a perturbed variable x_plus_delta
  /// @param[in] x_plus_delta Perturbed variable.
  /// @param[in] x Variable.
  /// @param[out] delta minimal difference.
  /// \return True on success.
  static bool minus(const double* x_plus_delta, const double* x, double* delta);

  /// \brief Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double* x, double* jacobian);

  /// \brief The parameter block dimension.
  virtual int GlobalSize() const {
    return 7;
  }

  /// \brief The parameter block local dimension.
  virtual int LocalSize() const {
    return 6;
  }

  int AmbientSize() const final {
    return 7;
  }

  int TangentSize() const final {
    return 6;
  }

  bool PlusJacobian(const double* x, double* jacobian) const final {
    return ComputeJacobian(x, jacobian);
  }
  
  bool MinusJacobian(const double* x, double* jacobian) const final {
    return liftJacobian(x, jacobian);
  }

  // added convenient check
  bool VerifyJacobianNumDiff(const double* x, double* jacobian,
                             double* jacobianNumDiff);
};

} // namespace ceres
#endif // POSE_MANIFOLD_H_
