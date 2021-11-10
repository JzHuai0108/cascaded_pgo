#ifndef HAMILTON_QUATERNION_H
#define HAMILTON_QUATERNION_H

#include <Eigen/Geometry>
#include <stdexcept>
// #include <sm/assert_macros.hpp>
// #include <sm/kinematics/rotations.hpp>

/**
 * Hamilton quaternion operators implemented following
 * Sola Quaternion Kinematics ...
 * The difference is that the layout [x,y,z,w] is used instead of [w,x,y,z] in Sola.
 */
namespace hamilton {
  template<typename T>
Eigen::Matrix<T, 3, 3> crossMx(const Eigen::Matrix<T, 3, 1>& vec)
{
    return (Eigen::Matrix<T, 3, 3>() << T(0), -vec(2), vec(1),
                                        vec(2), T(0), -vec(0),
                                        -vec(1), vec(0), T(0)).finished();
}


/// \brief Plus matrix of a quaternion, i.e. q_AB*q_BC =
/// plus(q_AB)*q_BC.coeffs().
/// @param[in] q_AB A Quaternion.
Eigen::Matrix4d quatPlus(const Eigen::Vector4d &q_AB);

/// \brief Oplus matrix of a quaternion, i.e. q_AB*q_BC =
/// oplus(q_BC)*q_AB.coeffs().
/// @param[in] q_BC A Quaternion.
Eigen::Matrix4d quatOPlus(const Eigen::Vector4d &q_BC);

template <typename Scalar_ = double>
inline Eigen::Matrix<Scalar_, 4, 1> quatIdentity() {
  return Eigen::Matrix<Scalar_, 4, 1>(Scalar_(0), Scalar_(0), Scalar_(0),
                                      Scalar_(1));
}

/**
 * @brief qplus(q, p) = quatPlus(q) * p
 * @param q
 * @param p
 * @return
 */
Eigen::Vector4d qplus(Eigen::Vector4d const &q, Eigen::Vector4d const &p);

Eigen::Vector4d qplus2(Eigen::Vector4d const &q, Eigen::Vector4d const &p);

Eigen::Matrix3d quat2r(Eigen::Vector4d const & q);

Eigen::Vector4d quatInv(Eigen::Vector4d const & q);
void invertQuat(Eigen::Vector4d & q);
Eigen::Vector3d qeps(Eigen::Vector4d const & q);
Eigen::Vector3f qeps(Eigen::Vector4f const & q);

double qeta(Eigen::Vector4d const & q);
float qeta(Eigen::Vector4f const & q);

template <typename Scalar_ = double>
inline bool isLessThenEpsilons4thRoot(Scalar_ x){
  static const Scalar_ epsilon4thRoot = pow(std::numeric_limits<Scalar_>::epsilon(), 1.0/4.0);
  return x < epsilon4thRoot;
}

Eigen::Vector4d axisAngle2quat(Eigen::Vector3d const & a);

/**
 * calculate arcsin(x)/x
 * @param x
 * @return
 */
template <typename Scalar_>
inline Scalar_ arcSinXOverX(Scalar_ x) {
  if(isLessThenEpsilons4thRoot(fabs(x))){
    return Scalar_(1.0) + x * x * Scalar_(1/6);
  }
  return asin(x) / x;
}


template <typename Scalar_>
Eigen::Matrix<Scalar_, 3, 3> quat2r(Eigen::Matrix<Scalar_, 4, 1> const &q) {
  Eigen::Map<const Eigen::Quaternion<Scalar_>> mq(q.data());
  return mq.template toRotationMatrix();
}

template <typename Scalar_>
Eigen::Matrix<Scalar_, 3, 1> quatRotate(Eigen::Matrix<Scalar_, 4, 1> const &q, Eigen::Matrix<Scalar_, 3, 1> const &v) {
  Eigen::Map<const Eigen::Quaternion<Scalar_>> mq(q.data());
  return mq * v;
}

template <typename Scalar_>
Eigen::Matrix<Scalar_, 3, 1> quat2AxisAngle(Eigen::Matrix<Scalar_, 4, 1> const & q)
{
  // SM_ASSERT_LT_DBG(std::runtime_error, fabs(q.norm() - 1), 8 * std::numeric_limits<Scalar_>::epsilon(), "This function is intended for unit quaternions only.");
  const Eigen::Matrix<Scalar_, 3, 1> a = qeps(q);
  const Scalar_ na = a.norm(), eta = qeta(q);
  Scalar_ scale;
  if(fabs(eta) < na){ // use eta because it is more precise than na to calculate the scale. No singularities here.
    scale = acos(eta) / na;
  } else {
    /*
     * In this case more precision is in na than in eta so lets use na only to calculate the scale:
     *
     * assume first eta > 0 and 1 > na > 0.
     *               u = asin (na) / na  (this implies u in [1, pi/2], because na i in [0, 1]
     *    sin (u * na) = na
     *  sin^2 (u * na) = na^2
     *  cos^2 (u * na) = 1 - na^2
     *                              (1 = ||q|| = eta^2 + na^2)
     *    cos^2 (u * na) = eta^2
     *                              (eta > 0,  u * na = asin(na) in [0, pi/2] => cos(u * na) >= 0 )
     *      cos (u * na) = eta
     *                              (u * na in [ 0, pi/2] )
     *                 u = acos (eta) / na
     *
     * So the for eta > 0 it is acos(eta) / na == asin(na) / na.
     * From some geometric considerations (mirror the setting at the hyper plane q==0) it follows for eta < 0 that (pi - asin(na)) / na = acos(eta) / na.
     */
    if(eta > 0){
      // For asin(na)/ na the singularity na == 0 can be removed. We can ask (e.g. Wolfram alpha) for its series expansion at na = 0. And that is done in the following function.
      scale = arcSinXOverX(na);
    }else{
      // (pi - asin(na))/ na has a pole at na == 0. So we cannot remove this singularity.
      // It is just the cut locus of the unit quaternion manifold at identity and thus the axis angle description becomes necessarily unstable there.
      scale = (M_PI - asin(na)) / na;
    }
  }
  return a * (Scalar_(2) * scale);
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
template <typename Scalar_>
Eigen::Matrix<Scalar_, 3, 3>
expDiffMat(const Eigen::Matrix<Scalar_, 3, 1> &vec) {
  Scalar_ phi = vec.norm();

  if (phi == 0) {
    return Eigen::Matrix<Scalar_, 3, 3>::Identity();
  }

  Eigen::Matrix<Scalar_, 3, 3> vecCross = crossMx(vec);

  Scalar_ phiAbs = fabs(phi);
  Scalar_ phiSquare = phi * phi;

  Scalar_ a;
  Scalar_ b;
  if (!isLessThenEpsilons4thRoot(phiAbs)) {
    Scalar_ siPhiHalf = sin(phi / 2);
    a = (2 * siPhiHalf * siPhiHalf / phiSquare);
    b = ((1 - sin(phi) / phi) / phiSquare);
  } else {
    a = (1.0 / 2) * (1 - (1.0 / (24 / 2)) * phiSquare);
    b = (1.0 / 6) * (1 - (1.0 / (120 / 6)) * phiSquare);
  }

  return Eigen::Matrix<Scalar_, 3, 3>::Identity() + a * vecCross +
         b * vecCross * vecCross;
}

/**
 * inverse of the left Jacobian of the Hamilton quaternion.
 * property 1. logDiffMat(vec) of Hamilton = logDiffMat(-vec) of JPL.
 * property 2. rightJacobianInvSO3(vec) = leftJacobianInvSO3(-vec).
 * \f$ f(X) = log(X) \f$
 * \f$ \frac{d\log(X)}{d\delta \phi} = \frac{d\log(\exp(\delta \phi)X)}{d\delta
 * \phi} \Big\vert_{\delta\phi = 0} = \mathbf{J}_l^{-1}(\log(X)) \f$
 */
template <typename Scalar_>
Eigen::Matrix<Scalar_, 3, 3>
logDiffMat(const Eigen::Matrix<Scalar_, 3, 1> &vec) {
  Scalar_ phi = vec.norm();
  if (phi == 0) {
    return Eigen::Matrix<Scalar_, 3, 3>::Identity();
  }

  Scalar_ phiAbs = fabs(phi);
  Eigen::Matrix<Scalar_, 3, 3> vecCross = crossMx(vec);

  Scalar_ a;
  if (!isLessThenEpsilons4thRoot(phiAbs)) {
    Scalar_ phiHalf = 0.5 * phi;
    a = ((1 - phiHalf / tan(phiHalf)) / phi / phi);
  } else {
    a = 1.0 / 12 * (1 + 1.0 / 60 * phi * phi);
  }
  return Eigen::Matrix<Scalar_, 3, 3>::Identity() - 0.5 * vecCross +
         a * vecCross * vecCross;
}

template <class Scalar>
struct Constants {
  static Scalar epsilon() { return Scalar(1e-10); }

  static Scalar epsilonSqrt() {
    using std::sqrt;
    return sqrt(epsilon());
  }

  static Scalar pi() {
    return Scalar(3.141592653589793238462643383279502884);
  }
};

/// Returns derivative of exp(x) wrt. x_i at x=0.
///
template <typename Scalar>
Eigen::Matrix<Scalar, 4, 3>
Dx_exp_x_at_0() {
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
template <typename Scalar>
Eigen::Matrix<Scalar, 4, 3> Dx_exp_x(
    Eigen::Matrix<Scalar, 3, 1> const& omega) {
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

} // namespace hamilton

#endif // HAMILTON_QUATERNION_H
