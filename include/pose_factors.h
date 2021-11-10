#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>

class EdgeSE3 : public ceres::SizedCostFunction<6, 7, 7> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSE3(const Eigen::Matrix<double, 7, 1> &T_ab)
      : p_ab_(Eigen::Map<const Eigen::Vector3d>(T_ab.data())),
        q_ab_(Eigen::Map<const Eigen::Quaterniond>(T_ab.data() + 3)) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d p_ab_;
  Eigen::Quaterniond q_ab_;
};

class EdgeSE3Prior : public ceres::SizedCostFunction<6, 7> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeSE3Prior(const Eigen::Matrix<double, 7, 1> &T)
      : p_(Eigen::Map<const Eigen::Vector3d>(T.data())),
        q_(Eigen::Map<const Eigen::Quaterniond>(T.data() + 3)) {}
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;
  Eigen::Vector3d p_;
  Eigen::Quaterniond q_;
};
