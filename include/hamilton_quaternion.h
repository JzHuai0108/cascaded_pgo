#ifndef HAMILTON_QUATERNION_H
#define HAMILTON_QUATERNION_H

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>


namespace Sophus {

template <class Scalar>
struct Constants {
  static Scalar epsilon() { return Scalar(1e-10); }

  static Scalar epsilonPlus() {
    return epsilon() * (Scalar(1.) + epsilon());
  }

  static Scalar epsilonSqrt() {
    using std::sqrt;
    return sqrt(epsilon());
  }

  static Scalar pi() {
    return Scalar(3.141592653589793238462643383279502884);
  }
};

template <>
struct Constants<float> {
  static float constexpr epsilon() {
    return static_cast<float>(1e-5);
  }
  static float epsilonPlus() {
    return epsilon() * (1.f + epsilon());
  }

  static float epsilonSqrt() { return std::sqrt(epsilon()); }

  static float constexpr pi() {
    return 3.141592653589793238462643383279502884f;
  }
};
}  // namespace Sophus

/**
 * Hamilton quaternion operators implemented following
 * Sola Quaternion Kinematics ...
 * The difference is that the layout [x,y,z,w] is used instead of [w,x,y,z] in
 * Sola.
 */
namespace hamilton {

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> crossMx(const Eigen::MatrixBase<Derived>& vec)
{
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  typedef typename Derived::Scalar T;
  return (Eigen::Matrix<T, 3, 3>() << T(0), -vec(2), vec(1),
                                        vec(2), T(0), -vec(0),
                                        -vec(1), vec(0), T(0)).finished();
}


template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3>
skew(const Eigen::MatrixBase<Derived> &vec) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  typedef typename Derived::Scalar T;
  return (Eigen::Matrix<T, 3, 3>() << T(0), -vec(2), vec(1), vec(2), T(0),
          -vec(0), -vec(1), vec(0), T(0))
      .finished();
}

template <typename Scalar = double>
inline Eigen::Matrix<Scalar, 4, 1> quatIdentity() {
  return Eigen::Matrix<Scalar, 4, 1>(Scalar(0), Scalar(0), Scalar(0),
                                      Scalar(1));
}

/**
 * @brief qplus(q, p) = quatPlus(q) * p
 * @param q
 * @param p
 * @return
 */

/// \brief Plus matrix of a quaternion, i.e. q_AB*q_BC =
/// plus(q_AB)*q_BC.coeffs().
/// @param[in] q_AB A Quaternion.
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4>
quatPlus(const Eigen::MatrixBase<Derived> &q) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  Eigen::Matrix<typename Derived::Scalar, 4, 4> Q;
  Q(0, 0) = q[3];
  Q(0, 1) = -q[2];
  Q(0, 2) = q[1];
  Q(0, 3) = q[0];
  Q(1, 0) = q[2];
  Q(1, 1) = q[3];
  Q(1, 2) = -q[0];
  Q(1, 3) = q[1];
  Q(2, 0) = -q[1];
  Q(2, 1) = q[0];
  Q(2, 2) = q[3];
  Q(2, 3) = q[2];
  Q(3, 0) = -q[0];
  Q(3, 1) = -q[1];
  Q(3, 2) = -q[2];
  Q(3, 3) = q[3];
  return Q;
}

/// \brief Oplus matrix of a quaternion, i.e. q_AB*q_BC =
/// oplus(q_BC)*q_AB.coeffs().
/// @param[in] q_BC A Quaternion.
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4>
quatOPlus(const Eigen::MatrixBase<Derived> &q) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  Eigen::Matrix<typename Derived::Scalar, 4, 4> Q;
  Q(0, 0) = q[3];
  Q(0, 1) = q[2];
  Q(0, 2) = -q[1];
  Q(0, 3) = q[0];
  Q(1, 0) = -q[2];
  Q(1, 1) = q[3];
  Q(1, 2) = q[0];
  Q(1, 3) = q[1];
  Q(2, 0) = q[1];
  Q(2, 1) = -q[0];
  Q(2, 2) = q[3];
  Q(2, 3) = q[2];
  Q(3, 0) = -q[0];
  Q(3, 1) = -q[1];
  Q(3, 2) = -q[2];
  Q(3, 3) = q[3];
  return Q;
}

template <typename Derived1, typename Derived2>
inline Eigen::Matrix<typename Derived1::Scalar, 4, 1>
qplus(const Eigen::MatrixBase<Derived1> &q,
      const Eigen::MatrixBase<Derived2> &p) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived1, 4, 1);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 4, 1);
  static_assert(
      std::is_same<typename Derived1::Scalar, typename Derived2::Scalar>::value,
      "You mixed different numeric types.");
  Eigen::Matrix<typename Derived1::Scalar, 4, 1> qplus_p;
  qplus_p[0] = q[3] * p[0] + q[0] * p[3] + q[1] * p[2] - q[2] * p[1];
  qplus_p[1] = q[3] * p[1] - q[0] * p[2] + q[1] * p[3] + q[2] * p[0];
  qplus_p[2] = q[3] * p[2] + q[0] * p[1] - q[1] * p[0] + q[2] * p[3];
  qplus_p[3] = q[3] * p[3] - q[0] * p[0] - q[1] * p[1] - q[2] * p[2];
  return qplus_p;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 1>
qplus2(const Eigen::MatrixBase<Derived> &q,
       const Eigen::MatrixBase<Derived> &p) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  Eigen::Map<const Eigen::Quaternion<typename Derived::Scalar>> sq(q.derived().data());
  Eigen::Map<const Eigen::Quaternion<typename Derived::Scalar>> sp(p.derived().data());
  Eigen::Matrix<typename Derived::Scalar, 4, 1> qplus_p;
  Eigen::Map<Eigen::Quaternion<typename Derived::Scalar>> qp(qplus_p.data());
  qp = sq * sp;
  return qplus_p;
}

template <typename Derived>
inline void invertQuat(Eigen::MatrixBase<Derived> &q) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  Eigen::Map<Eigen::Quaternion<typename Derived::Scalar>> eq(
      q.derived().data());
  eq = eq.conjugate();
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 1>
quatInv(const Eigen::MatrixBase<Derived> &q) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  Eigen::Matrix<typename Derived::Scalar, 4, 1> qret = q;
  invertQuat(qret);
  return qret;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 1>
qeps(const Eigen::MatrixBase<Derived> &q) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  return q.template head<3>();
}

template <typename Derived>
inline typename Derived::Scalar qeta(const Eigen::MatrixBase<Derived> &q) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  return q[3];
}

template <typename T> inline T jetfabs(const T &f) {
  return f < T(0.0) ? -f : f;
}

template <typename Scalar = double>
inline bool isLessThenEpsilons4thRoot(Scalar x) {
  static const Scalar epsilon4thRoot =
      pow(std::numeric_limits<Scalar>::epsilon(), 1.0 / 4.0);
  return x < epsilon4thRoot;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 1>
axisAngle2quat(const Eigen::MatrixBase<Derived> &omega) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  typedef typename Derived::Scalar Scalar;
#if 1
  using std::abs;
  using std::cos;
  using std::sin;
  using std::sqrt;
  Scalar angle;
  Scalar *theta = &angle;

  Scalar theta_sq = omega.squaredNorm();

  Scalar imag_factor;
  Scalar real_factor;
  if (theta_sq <
          Sophus::Constants<Scalar>::epsilon() * Sophus::Constants<Scalar>::epsilon()) {
          *theta = Scalar(0);
          Scalar theta_po4 = theta_sq * theta_sq;
          imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq +
                  Scalar(1.0 / 3840.0) * theta_po4;
          real_factor = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq +
                  Scalar(1.0 / 384.0) * theta_po4;
      } else {
          *theta = sqrt(theta_sq);
          Scalar half_theta = Scalar(0.5) * (*theta);
          Scalar sin_half_theta = sin(half_theta);
          imag_factor = sin_half_theta / (*theta);
          real_factor = cos(half_theta);
      }
  return Eigen::Matrix<Scalar, 4, 1>(imag_factor * omega.x(),
                                     imag_factor * omega.y(), imag_factor * omega.z(), real_factor);

#else
  // jhuai: The below implementation causes ceres Jet NaN when omega are zeros.
  // Method of implementing this function that is accurate to numerical
  // precision from Grassia, F. S. (1998). Practical parameterization of
  // rotations using the exponential map. journal of graphics, gpu, and game
  // tools, 3(3):29–48.
  Scalar theta = omega.norm();

  // na is 1/theta sin(theta/2)
  Scalar na;
  if (isLessThenEpsilons4thRoot(theta)) {
    static const Scalar one_over_48 = Scalar(1.0 / 48.0);
    na = Scalar(0.5) + (theta * theta) * one_over_48;
  } else {
    na = sin(theta * Scalar(0.5)) / theta;
  }
  Eigen::Matrix<Scalar, 3, 1> axis = omega * na;
  Scalar ct = cos(theta * Scalar(0.5));
  return Eigen::Matrix<Scalar, 4, 1>(axis[0], axis[1], axis[2], ct);
#endif
}

/**
 * calculate arcsin(x)/x
 * @param x
 * @return
 */
template <typename Scalar> inline Scalar arcSinXOverX(Scalar x) {
  if (isLessThenEpsilons4thRoot(jetfabs(x))) {
    // return Scalar(1.0) + x * x * Scalar(1.0 / 6); // This line will cause infinite derivatives when x == 0.
    return Scalar(1.0);
  }
  return asin(x) / x;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3>
quat2r(const Eigen::MatrixBase<Derived> &q) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  Eigen::Map<const Eigen::Quaternion<typename Derived::Scalar>> mq(q.derived().data());
  return mq.template toRotationMatrix();
}

template <typename DerivedQ, typename DerivedV>
inline Eigen::Matrix<typename DerivedQ::Scalar, 3, 1>
quatRotate(const Eigen::MatrixBase<DerivedQ> &q,
           const Eigen::MatrixBase<DerivedV> &v) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DerivedQ, 4, 1);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DerivedV, 3, 1);
  static_assert(
      std::is_same<typename DerivedQ::Scalar, typename DerivedV::Scalar>::value,
      "You mixed different numeric types.");
  Eigen::Map<const Eigen::Quaternion<typename DerivedQ::Scalar>> mq(
      q.derived().data());
  return mq * v;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1>
quat2AxisAngle(const Eigen::MatrixBase<Derived> &q) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 4, 1);
  typedef typename Derived::Scalar Scalar;
#if 0
  // jhuai: The sophus implementation fails tests SO3BSplineTestSuite.SplineEvalRiDJacobianTesterD0,1,2.
  Eigen::Matrix<Scalar, 3, 1> tangent;
  Scalar theta;
  using std::abs;
  using std::atan2;
  using std::sqrt;
  Scalar squared_n = qeps(q).squaredNorm();
  Scalar w = qeta(q);

  Scalar two_atan_nbyw_by_n;

  /// Atan-based log thanks to
  ///
  /// C. Hertzberg et al.:
  /// "Integrating Generic Sensor Fusion Algorithms with Sound State
  /// Representation through Encapsulation of Manifolds"
  /// Information Fusion, 2011

  if (squared_n <
      Sophus::Constants<Scalar>::epsilon() * Sophus::Constants<Scalar>::epsilon()) {
    // If quaternion is normalized and n=0, then w should be 1;
    // w=0 should never happen here!
    Scalar squared_w = w * w;
    two_atan_nbyw_by_n =
        Scalar(2) / w - Scalar(2.0 / 3.0) * (squared_n) / (w * squared_w);
    theta = Scalar(2) * squared_n / w;
  } else {
    Scalar n = sqrt(squared_n);

    // w < 0 ==> cos(theta/2) < 0 ==> theta > pi
    //
    // By convention, the condition |theta| < pi is imposed by wrapping theta
    // to pi; The wrap operation can be folded inside evaluation of atan2
    //
    // theta - pi = atan(sin(theta - pi), cos(theta - pi))
    //            = atan(-sin(theta), -cos(theta))
    //
    Scalar atan_nbyw =
        (w < Scalar(0)) ? Scalar(atan2(-n, -w)) : Scalar(atan2(n, w));
    two_atan_nbyw_by_n = Scalar(2) * atan_nbyw / n;
    theta = two_atan_nbyw_by_n * n;
  }
  tangent = two_atan_nbyw_by_n * qeps(q);
  return tangent;
#else
  Scalar qnorm = q.template norm();
  Scalar diff = qnorm - Scalar(1);
  Scalar absdiff = jetfabs(diff);
  if (absdiff >= Scalar(1000) * std::numeric_limits<Scalar>::epsilon()) {
      std::ostringstream oss;
      oss << "This function is intended for unit quaternions only. "
          << "absdiff: " << absdiff
          << ", threshold: " << (Scalar(1000) * std::numeric_limits<Scalar>::epsilon());
      throw std::runtime_error(oss.str());
  }
  const Eigen::Matrix<Scalar, 3, 1> a = qeps(q);
  const Scalar na = a.norm(); // This will cause infinity derivatives if na is 0.
  const Scalar eta = qeta(q);
  Scalar scale;
  if (jetfabs(eta) < na) { // use eta because it is more precise than na to
                           // calculate the scale. No singularities here.
    scale = acos(eta) / na;
  } else {
    /*
     * In this case more precision is in na than in eta so lets use na only to
     * calculate the scale:
     *
     * assume first eta > 0 and 1 > na > 0.
     *               u = asin (na) / na  (this implies u in [1, pi/2], because
     * na i in [0, 1] sin (u * na) = na sin^2 (u * na) = na^2 cos^2 (u * na) = 1
     * - na^2 (1 = ||q|| = eta^2 + na^2) cos^2 (u * na) = eta^2 (eta > 0,  u *
     * na = asin(na) in [0, pi/2] => cos(u * na) >= 0 ) cos (u * na) = eta (u *
     * na in [ 0, pi/2] ) u = acos (eta) / na
     *
     * So the for eta > 0 it is acos(eta) / na == asin(na) / na.
     * From some geometric considerations (mirror the setting at the hyper plane
     * q==0) it follows for eta < 0 that (pi - asin(na)) / na = acos(eta) / na.
     */
    if (eta > 0) {
      // For asin(na)/ na the singularity na == 0 can be removed. We can ask
      // (e.g. Wolfram alpha) for its series expansion at na = 0. And that is
      // done in the following function.
      scale = arcSinXOverX(na);
    } else {
      // (pi - asin(na))/ na has a pole at na == 0. So we cannot remove this
      // singularity. It is just the cut locus of the unit quaternion manifold
      // at identity and thus the axis angle description becomes necessarily
      // unstable there.
      scale = (M_PI - asin(na)) / na;
    }
  }
  return a * (Scalar(2) * scale);
#endif
}

/**
 * left Jacobian of the Hamilton quaternion.
 * property 1. expDiffMat(vec) of Hamilton = expDiffMat(-vec) of JPL.
 * property 2. rightJacobianSO3(vec) = leftJacobianSO3(-vec).
 * \f$ f(\phi) = \exp(\phi) = X \f$
 * Perturbation in X is defined to be \f$ \exp(\delta \phi) X = \hat{X} \f$.
 * \f$ \exp(\mathbf{J}_l(\phi) \delta \phi) \exp(\phi) = \exp(\phi + \delta
 * \phi) \f$
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
expDiffMat(const Eigen::MatrixBase<Derived> &vec) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  typedef typename Derived::Scalar Scalar;
  Scalar phi = vec.norm();

  if (phi == 0) {
    return Eigen::Matrix<Scalar, 3, 3>::Identity();
  }

  Eigen::Matrix<Scalar, 3, 3> vecCross = skew(vec);

  Scalar phiAbs = fabs(phi);
  Scalar phiSquare = phi * phi;

  Scalar a;
  Scalar b;
  if (!isLessThenEpsilons4thRoot(phiAbs)) {
    Scalar siPhiHalf = sin(phi / 2);
    a = (2 * siPhiHalf * siPhiHalf / phiSquare);
    b = ((1 - sin(phi) / phi) / phiSquare);
  } else {
    a = (1.0 / 2) * (1 - (1.0 / (24 / 2)) * phiSquare);
    b = (1.0 / 6) * (1 - (1.0 / (120 / 6)) * phiSquare);
  }

  return Eigen::Matrix<Scalar, 3, 3>::Identity() + a * vecCross +
         b * vecCross * vecCross;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
leftJacobian(const Eigen::MatrixBase<Derived> &vec) {
  return expDiffMat(vec); 
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
rightJacobian(const Eigen::MatrixBase<Derived> &vec) {
  return expDiffMat(-vec); 
}

/**
 * inverse of the left Jacobian of the Hamilton quaternion.
 * property 1. logDiffMat(vec) of Hamilton = logDiffMat(-vec) of JPL.
 * property 2. rightJacobianInvSO3(vec) = leftJacobianInvSO3(-vec).
 * \f$ f(X) = log(X) \f$
 * \f$ \frac{d\log(X)}{d\delta \phi} = \frac{d\log(\exp(\delta \phi)X)}{d\delta
 * \phi} \Big\vert_{\delta\phi = 0} = \mathbf{J}_l^{-1}(\log(X)) \f$
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
logDiffMat(const Eigen::MatrixBase<Derived> &vec) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  typedef typename Derived::Scalar Scalar;
  Scalar phi = vec.norm();
  if (phi == Scalar(0)) {
    return Eigen::Matrix<Scalar, 3, 3>::Identity();
  }

  Scalar phiAbs = fabs(phi);
  Eigen::Matrix<Scalar, 3, 3> vecCross = skew(vec);

  Scalar a;
  if (!isLessThenEpsilons4thRoot(phiAbs)) {
    Scalar phiHalf = Scalar(0.5) * phi;
    a = ((Scalar(1) - phiHalf / tan(phiHalf)) / phi / phi);
  } else {
    a = Scalar(1.0 / 12) * (Scalar(1) + Scalar(1.0 / 60) * phi * phi);
  }
  return Eigen::Matrix<Scalar, 3, 3>::Identity() - Scalar(0.5) * vecCross +
         a * vecCross * vecCross;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
invLeftJacobian(const Eigen::MatrixBase<Derived> &vec) {
  return logDiffMat(vec); 
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
invRightJacobian(const Eigen::MatrixBase<Derived> &vec) {
  return logDiffMat(-vec); 
}

template <class Scalar> struct Constants {
  static Scalar epsilon() { return Scalar(1e-10); }

  static Scalar epsilonSqrt() {
    using std::sqrt;
    return sqrt(epsilon());
  }

  static Scalar pi() { return Scalar(3.141592653589793238462643383279502884); }
};

/// Returns derivative of exp(x) wrt. x_i at x=0.
///
template <typename Scalar> Eigen::Matrix<Scalar, 4, 3> Dx_exp_x_at_0() {
  Eigen::Matrix<Scalar, 4, 3> J;
  // clang-format off
  J <<  Scalar(0.5),   Scalar(0),   Scalar(0),
          Scalar(0), Scalar(0.5),   Scalar(0),
          Scalar(0),   Scalar(0), Scalar(0.5),
          Scalar(0),   Scalar(0),   Scalar(0);
  // clang-format on
  return J;
}

/// Returns derivative of exp(x) wrt. x.
///
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 3>
Dx_exp_x(const Eigen::MatrixBase<Derived> &omega) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  typedef typename Derived::Scalar Scalar;
  using std::cos;
  using std::exp;
  using std::sin;
  using std::sqrt;
  Scalar const c0 = omega[0] * omega[0];
  Scalar const c1 = omega[1] * omega[1];
  Scalar const c2 = omega[2] * omega[2];
  Scalar const c3 = c0 + c1 + c2;

  if (c3 < Constants<Scalar>::epsilon()) {
    return Dx_exp_x_at_0<Scalar>();
  }

  Scalar const c4 = sqrt(c3);
  Scalar const c5 = 1.0 / c4;
  Scalar const c6 = 0.5 * c4;
  Scalar const c7 = sin(c6);
  Scalar const c8 = c5 * c7;
  Scalar const c9 = pow(c3, -3.0L / 2.0L);
  Scalar const c10 = c7 * c9;
  Scalar const c11 = Scalar(1.0) / c3;
  Scalar const c12 = cos(c6);
  Scalar const c13 = Scalar(0.5) * c11 * c12;
  Scalar const c14 = c7 * c9 * omega[0];
  Scalar const c15 = Scalar(0.5) * c11 * c12 * omega[0];
  Scalar const c16 = -c14 * omega[1] + c15 * omega[1];
  Scalar const c17 = -c14 * omega[2] + c15 * omega[2];
  Scalar const c18 = omega[1] * omega[2];
  Scalar const c19 = -c10 * c18 + c13 * c18;
  Scalar const c20 = Scalar(0.5) * c5 * c7;
  Eigen::Matrix<Scalar, 4, 3> J;
  J(0, 0) = -c0 * c10 + c0 * c13 + c8;
  J(0, 1) = c16;
  J(0, 2) = c17;
  J(1, 0) = c16;
  J(1, 1) = -c1 * c10 + c1 * c13 + c8;
  J(1, 2) = c19;
  J(2, 0) = c17;
  J(2, 1) = c19;
  J(2, 2) = -c10 * c2 + c13 * c2 + c8;
  J(3, 0) = -c20 * omega[0];
  J(3, 1) = -c20 * omega[1];
  J(3, 2) = -c20 * omega[2];
  return J;
}

template <typename T>
void ensureCloseQuaternion(std::vector<Eigen::Matrix<T, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 4, 1>>> &qlist) {
  for (size_t i = 1; i < qlist.size(); ++i) {
    if ((qlist[i] - qlist[i-1]).squaredNorm() > (qlist[i] + qlist[i-1]).squaredNorm()) {
      qlist[i] = -qlist[i];
    }
  }
}

template <typename T>
void ensureCloseQuaternion(const Eigen::Matrix<T, 4, 1> &q1, Eigen::Matrix<T, 4, 1> &q2) {
  if ((q1 - q2).squaredNorm() > (q1 + q2).squaredNorm()) {
    q2 = -q2;
  }
}

} // namespace hamilton

#endif // HAMILTON_QUATERNION_H
