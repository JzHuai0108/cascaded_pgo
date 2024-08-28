#ifndef FACTOR_GRAVITY_MANIFOLD_H_
#define FACTOR_GRAVITY_MANIFOLD_H_


#include <ceres/manifold.h>
#include <Eigen/Eigen>

#include <iostream>
#include <random>

namespace ceres {

typedef Eigen::Quaterniond QPD;
typedef Eigen::MatrixXd MXD;

/*!
 * \brief Gets a skew-symmetric matrix from a (column) vector
 * \param   vec 3x1-matrix (column vector)
 * \return skew   3x3-matrix
 */

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3>
crossMat(const Eigen::MatrixBase<Derived> &vec) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  typedef typename Derived::Scalar T;
  return (Eigen::Matrix<T, 3, 3>() << T(0), -vec(2), vec(1), vec(2), T(0),
          -vec(0), -vec(1), vec(0), T(0))
      .finished();
}

template <typename DERIVED, typename GET, unsigned int D, unsigned int E = D>
class ElementBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ElementBase(){};
  virtual ~ElementBase(){};
  static const unsigned int D_ = D;
  static const unsigned int E_ = E;
  typedef Eigen::Matrix<double, D_, 1> mtDifVec;
  typedef GET mtGet;
  std::string name_;
  virtual void boxPlus(const mtDifVec& vecIn, DERIVED& stateOut) const = 0;
  virtual void boxMinus(const DERIVED& stateIn, mtDifVec& vecOut) const = 0;
  virtual void boxMinusJac(const DERIVED& stateIn, MXD& matOut) const = 0;
  virtual void print() const = 0;
  virtual void setIdentity() = 0;
  virtual void setRandom(unsigned int& s) = 0;
  virtual void fix() = 0;
  static DERIVED Identity() {
    DERIVED identity;
    identity.setIdentity();
    return identity;
  }
  DERIVED& operator=(DERIVED other) {
    other.swap(*this);
    return *this;
  }
  virtual mtGet& get(unsigned int i) = 0;
  virtual const mtGet& get(unsigned int i) const = 0;
};

class NormalVectorElement
    : public ElementBase<NormalVectorElement, NormalVectorElement, 2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Quaterniond q_;
  const Eigen::Vector3d e_x;
  const Eigen::Vector3d e_y;
  const Eigen::Vector3d e_z;
  NormalVectorElement() : e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {}
  NormalVectorElement(const NormalVectorElement& other)
      : e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {
    q_ = other.q_;
  }
  NormalVectorElement(const Eigen::Vector3d& vec)
      : e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {
    setFromVector(vec);
  }
  NormalVectorElement(const Eigen::Quaterniond& q)
      : e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {
    q_ = q;
  }
  NormalVectorElement(double w, double x, double y, double z)
      : q_(w, x, y, z), e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {}
  virtual ~NormalVectorElement(){};
  Eigen::Vector3d getVec() const { return q_ * e_z; }
  Eigen::Vector3d getPerp1() const { return q_ * e_x; }
  Eigen::Vector3d getPerp2() const { return q_ * e_y; }
  NormalVectorElement& operator=(const NormalVectorElement& other) {
    q_ = other.q_;
    return *this;
  }
  static Eigen::Vector3d getRotationFromTwoNormals(
      const Eigen::Vector3d& a, const Eigen::Vector3d& b,
      const Eigen::Vector3d& a_perp) {
    const Eigen::Vector3d cross = a.cross(b);
    const double crossNorm = cross.norm();
    const double c = a.dot(b);
    const double angle = std::acos(c);
    if (crossNorm < 1e-6) {
      if (c > 0) {
        return cross;
      } else {
        return a_perp * M_PI;
      }
    } else {
      return cross * (angle / crossNorm);
    }
  }
  static Eigen::Vector3d getRotationFromTwoNormals(
      const NormalVectorElement& a, const NormalVectorElement& b) {
    return getRotationFromTwoNormals(a.getVec(), b.getVec(), a.getPerp1());
  }
  static Eigen::Matrix3d getRotationFromTwoNormalsJac(
      const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const Eigen::Vector3d cross = a.cross(b);
    const double crossNorm = cross.norm();
    Eigen::Vector3d crossNormalized = cross / crossNorm;
    Eigen::Matrix3d crossNormalizedSqew = crossMat(crossNormalized);
    const double c = a.dot(b);
    const double angle = std::acos(c);
    if (crossNorm < 1e-6) {
      if (c > 0) {
        return -crossMat(b);
      } else {
        return Eigen::Matrix3d::Zero();
      }
    } else {
      return -1 / crossNorm *
             (crossNormalized * b.transpose() -
              (crossNormalizedSqew * crossNormalizedSqew * crossMat(b) *
               angle));
    }
  }
  static Eigen::Matrix3d getRotationFromTwoNormalsJac(
      const NormalVectorElement& a, const NormalVectorElement& b) {
    return getRotationFromTwoNormalsJac(a.getVec(), b.getVec());
  }
  void setFromVector(Eigen::Vector3d vec) {
    const double d = vec.norm();
    if (d > 1e-6) {
      vec = vec / d;
      Eigen::Vector3d rv = getRotationFromTwoNormals(e_z, vec, e_x);
      Eigen::AngleAxisd aa(rv.norm(), rv.normalized());
      q_ = Eigen::Quaterniond(aa);
    } else {
      q_.setIdentity();
    }
  }
  NormalVectorElement rotated(const QPD& q) const {
    return NormalVectorElement(q * q_);
  }
  NormalVectorElement inverted() const {
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI, getPerp1()));
    return NormalVectorElement(q * q_);
  }
  void boxPlus(const mtDifVec& vecIn, NormalVectorElement& stateOut) const {
    Eigen::Vector3d Nu = vecIn(0) * getPerp1() + vecIn(1) * getPerp2();
    Eigen::Quaterniond q(Eigen::AngleAxisd(Nu.norm(), Nu.normalized()));
    stateOut.q_ = q * q_;
  }
  void boxMinus(const NormalVectorElement& stateIn, mtDifVec& vecOut) const {
    vecOut =
        stateIn.getN().transpose() * getRotationFromTwoNormals(stateIn, *this);
  }
  void boxMinusJac(const NormalVectorElement& stateIn, MXD& matOut) const {
    matOut = -stateIn.getN().transpose() *
             getRotationFromTwoNormalsJac(*this, stateIn) * this->getM();
  }
  static mtDifVec boxMinus(const Eigen::Vector3d &l, const Eigen::Vector3d &r) {
    mtDifVec delta;
    NormalVectorElement(l).boxMinus(NormalVectorElement(r), delta);
    return delta;
  }
  static Eigen::Vector3d boxPlus(const Eigen::Vector3d &l, const mtDifVec &r) {
    NormalVectorElement s;
    NormalVectorElement(l).boxPlus(r, s);
    return s.getVec();
  }
  void print() const { std::cout << getVec().transpose() << std::endl; }
  void setIdentity() { q_.setIdentity(); }
  void setRandom(unsigned int& s) {
    std::default_random_engine generator(s);
    std::normal_distribution<double> distribution(0.0, 1.0);
    q_.w() = distribution(generator);
    q_.x() = distribution(generator);
    q_.y() = distribution(generator);
    q_.z() = distribution(generator);
    q_.normalize();
    s++;
  }
  void fix() { q_.normalize(); }
  mtGet& get(unsigned int /*i*/ = 0) {
    return *this;
  }
  const mtGet& get(unsigned int /*i*/ = 0) const {
    return *this;
  }
  // return dVec/dmininalRep
  Eigen::Matrix<double, 3, 2> getM() const {
    Eigen::Matrix<double, 3, 2> M;
    M.col(0) = -getPerp2();
    M.col(1) = getPerp1();
    return M;
  }
  Eigen::Matrix<double, 3, 2> getN() const {
    Eigen::Matrix<double, 3, 2> M;
    M.col(0) = getPerp1();
    M.col(1) = getPerp2();
    return M;
  }
  const double* data() const { return q_.coeffs().data(); }
};

class UnitVec3Manifold : public ::ceres::Manifold {
  public:
    static const int kGlobalDim = 3;
    static const int kLocalDim = 2;

    int AmbientSize() const final { return kGlobalDim; }
    int TangentSize() const final { return kLocalDim; }
    /**
     * @brief PlusJacobian
     * @param x
     * @param jacobian 3x2 row major matrix
     */
    bool PlusJacobian(const double *x, double *jacobian) const final {
      return plusJacobian(x, jacobian);
    }
    /**
     * @brief MinusJacobian
     * @param x
     * @param jacobian 2x3 row major matrix
     */
    bool MinusJacobian(const double *x, double *jacobian) const final {
      return liftJacobian(x, jacobian);
    }

    // Generalization of the addition operation,
    //
    //   x_plus_delta = Plus(x, delta)
    //
    // with the condition that Plus(x, 0) = x.
    virtual bool Plus(const double *x, const double *delta,
              double *x_plus_delta) const final {
      return plus(x, delta, x_plus_delta);
    }

    // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
    //
    // jacobian is a row-major GlobalSize() x LocalSize() matrix.
    virtual bool ComputeJacobian(const double *x, double *jacobian) const final {
      return plusJacobian(x, jacobian);
    }

    // Size of x.
    virtual int GlobalSize() const final { return kGlobalDim; }
    // Size of delta.
    virtual int LocalSize() const final { return kLocalDim; }

    /// \brief Trivial destructor.
    virtual ~UnitVec3Manifold() final {}

    /// \brief Computes the minimal difference between a variable x and a
    /// perturbed variable x_plus_delta
    /// @param[in] y Perturbed Variable.
    /// @param[in] x variable.
    /// @param[out] y_minus_x minimal difference.
    /// \return True on success.
    bool Minus(const double* y, const double* x, double* y_minus_x) const final {
      return minus(y, x, y_minus_x);
    }

    /// \brief Computes the Jacobian from minimal space to naively
    /// overparameterised space as used by ceres.
    /// @param[in] x Variable.
    /// @param[out] jacobian the Jacobian (dimension minDim x dim).
    /// \return True on success.
    virtual bool ComputeLiftJacobian(const double *x, double *jacobian) const final {
      return liftJacobian(x, jacobian);
    }

    static bool liftJacobian(const double *x0, double *jacobian) {
      Eigen::Map<const Eigen::Vector3d> vx(x0);
      NormalVectorElement nve(vx);
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> j(jacobian);
      j = nve.getM().transpose(); // This is the pinv of M.
      return true;
    }

    static bool minus(const double *x_plus_delta, const double *x,
                      double *delta) {
      Eigen::Map<const Eigen::Vector3d> vxplus(x_plus_delta);
      Eigen::Map<const Eigen::Vector3d> vx(x);
      Eigen::Map<Eigen::Vector2d> d(delta);
      d = NormalVectorElement::boxMinus(vxplus, vx);
      return true;
    }

    static bool plus(const double *x, const double *delta, double *x_plus_delta) {
      Eigen::Map<Eigen::Vector3d> xplus(x_plus_delta);
      xplus = NormalVectorElement::boxPlus(
          Eigen::Map<const Eigen::Vector3d>(x),
          Eigen::Map<const Eigen::Vector2d>(delta));
      return true;
    }

    /**
     * @brief plusJacobian
     * @param x0
     * @param jacobian  is a row-major 3 x 2 matrix.
     * @return
     */
    static bool plusJacobian(const double *x0, double *jacobian) {
      Eigen::Map<const Eigen::Vector3d> vx0(x0);
      NormalVectorElement nve0(vx0);
      Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> j(jacobian);
      j = nve0.getM();
      return true;
    }
  };
}  // namespace ceres
#endif // FACTOR_GRAVITY_MANIFOLD_H_
