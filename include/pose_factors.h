#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>

class EdgeSE3 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSE3(const Eigen::Matrix<double, 7, 1> &T_ab, const Eigen::Matrix<double, 6, 1> sigmaPQ)
      : p_ab_(Eigen::Map<const Eigen::Vector3d>(T_ab.data())),
        q_ab_(Eigen::Map<const Eigen::Quaterniond>(T_ab.data() + 3)), 
        sigmaPQ_(sigmaPQ) {}

  template <typename T>
  bool operator()(const T *const pq1, const T *const pq2, T *residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1(pq1);
    Eigen::Map<const Eigen::Quaternion<T>> q1(pq1 + 3);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p2(pq2);
    Eigen::Map<const Eigen::Quaternion<T>> q2(pq2 + 3);
    Eigen::Matrix<T, 3, 1> p12 = q1.inverse() * (p2 - p1);
    Eigen::Quaternion<T> q12 = q1.inverse() * q2;
    Eigen::Map<Eigen::Matrix<T, 6, 1>> error(residuals);
    error.template head<3>() = (p12 - p_ab_.cast<T>()).cwiseQuotient(sigmaPQ_.head<3>().cast<T>());
    Eigen::Quaternion<T> dq = q12 * q_ab_.inverse().template cast<T>();
    Eigen::AngleAxis<T> aa(dq);
    error.template tail<3>() = (aa.axis() * aa.angle()).cwiseQuotient(sigmaPQ_.tail<3>().cast<T>());
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Matrix<double, 7, 1> &T_ab, const Eigen::Matrix<double, 6, 1> sigmaPQ) {
    return new ceres::AutoDiffCostFunction<EdgeSE3, 6, 7, 7>(new EdgeSE3(T_ab, sigmaPQ));
  }

  Eigen::Vector3d p_ab_;
  Eigen::Quaterniond q_ab_;
  Eigen::Matrix<double, 6, 1> sigmaPQ_;
};

class EdgeSE3Prior {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSE3Prior(const Eigen::Matrix<double, 7, 1> &T, const Eigen::Matrix<double, 6, 1> sigmaPQ)
      : p_(Eigen::Map<const Eigen::Vector3d>(T.data())),
        q_(Eigen::Map<const Eigen::Quaterniond>(T.data() + 3)),
        sigmaPQ_(sigmaPQ) {}

  template <typename T>
  bool operator()(const T *const pq, T *residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(pq);
    Eigen::Map<const Eigen::Quaternion<T>> q(pq + 3);

    Eigen::Map<Eigen::Matrix<T, 6, 1>> error(residuals);
    Eigen::Matrix<T, 3, 1> p_err = (p - p_.cast<T>()).cwiseQuotient(sigmaPQ_.head<3>().cast<T>());
    Eigen::Quaternion<T> dq = q * q_.inverse().template cast<T>();
    Eigen::AngleAxis<T> aa(dq);
    Eigen::Matrix<T, 3, 1> q_err = (aa.axis() * aa.angle()).cwiseQuotient(sigmaPQ_.tail<3>().cast<T>());

    error << p_err, q_err;
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Matrix<double, 7, 1> &T, const Eigen::Matrix<double, 6, 1> sigmaPQ) {
    return new ceres::AutoDiffCostFunction<EdgeSE3Prior, 6, 7>(new EdgeSE3Prior(T, sigmaPQ));
  }

  Eigen::Vector3d p_;
  Eigen::Quaterniond q_;
  Eigen::Matrix<double, 6, 1> sigmaPQ_;
};
