#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <okvis/Time.hpp>

#include <ceres/cost_function.h>
#include <ceres/autodiff_cost_function.h>

struct GnssPosition {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::Time time;
    Eigen::Vector3d position; // E_p_B
    Eigen::Quaterniond rotation; // E_q_B
    Eigen::Vector3d sigma;
    int status; // NavSatFix status

    GnssPosition(const okvis::Time &t, const Eigen::Vector3d &p, 
                 const Eigen::Quaterniond &r, const Eigen::Vector3d &s, int st) : 
        time(t), position(p), rotation(r), sigma(s), status(st) {}

    friend std::ostream &operator<<(std::ostream &os, const GnssPosition &gp) {
        os << gp.time << " " << gp.position.transpose() 
           << " " << gp.rotation.coeffs().transpose() << " " << gp.sigma.transpose() << " " << gp.status;
        return os;
    }

    bool operator<(const GnssPosition &gp) const {
        return time < gp.time;
    }
};

typedef std::vector<GnssPosition, Eigen::aligned_allocator<GnssPosition>> PositionVector;

class SO3Prior {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SO3Prior(const Eigen::Quaterniond &q, const Eigen::Vector3d &sigma_q) : q_(q), sigma_q_(sigma_q) {
  }

  template <typename T>
  bool operator()(const T *const q, T *residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> q_map(q);

    Eigen::Quaternion<T> q_conj = q_.cast<T>().inverse();
    Eigen::Quaternion<T> q_diff = q_conj * q_map;

    Eigen::AngleAxis<T> aa(q_diff);
    Eigen::Matrix<T, 3, 1> angle_axis = aa.angle() * aa.axis();

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residual);
    residual_map= angle_axis.cwiseQuotient(sigma_q_.template cast<T>());
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Quaterniond &q, const Eigen::Vector3d &sigma_q) {
    return new ceres::AutoDiffCostFunction<SO3Prior, 3, 4>(new SO3Prior(q, sigma_q));
  }

private:
    Eigen::Quaterniond q_;
    Eigen::Vector3d sigma_q_;
};

// note the lidar frame is left back up wrt to the vehicle.
class NonHolonimicConstraint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    NonHolonimicConstraint(const double dt, const Eigen::Vector2d &sigma_v_yz)
        : dt_(dt), sigma_v_yz_(sigma_v_yz) {}

    template <typename T>
    bool operator()(const T *const W_T_L1, const T *const W_T_L2, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> W_p_L1(W_T_L1);
        Eigen::Map<const Eigen::Quaternion<T>> W_q_L1(W_T_L1 + 3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> W_p_L2(W_T_L2);
        Eigen::Map<const Eigen::Quaternion<T>> W_q_L2(W_T_L2 + 3);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residual_map(residual);

        Eigen::Matrix<T, 3, 1> W_v_L1 = (W_p_L2 - W_p_L1) / T(dt_);
        Eigen::Matrix<T, 3, 1> L_v1 = W_q_L1.conjugate() * W_v_L1;
        residual_map[0] = L_v1[0] / sigma_v_yz_[0]; // left
        residual_map[1] = L_v1[2] / sigma_v_yz_[1]; // up
        return true;
    }

    static ceres::CostFunction *Create(const double dt, const Eigen::Vector2d &sigma_v_yz) {
        return new ceres::AutoDiffCostFunction<NonHolonimicConstraint, 2, 7, 7>(
            new NonHolonimicConstraint(dt, sigma_v_yz));
    }

private:
    double dt_;
    Eigen::Vector2d sigma_v_yz_;
};

class SO3Edge {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SO3Edge(const Eigen::Quaterniond &b1_q_b2, const Eigen::Vector3d &sigma_q) : b1_q_b2_(b1_q_b2), sigma_q_(sigma_q) {}

  template <typename T>
  bool operator()(const T *const q1, const T *const q2, T *residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> q1_map(q1);
    Eigen::Map<const Eigen::Quaternion<T>> q2_map(q2);

    Eigen::Quaternion<T> q1_inv = q1_map.conjugate();
    Eigen::Quaternion<T> q_diff = q1_inv * q2_map * b1_q_b2_.template cast<T>().conjugate();

    Eigen::AngleAxis<T> aa(q_diff);
    Eigen::Matrix<T, 3, 1> angle_axis = aa.angle() * aa.axis();

    Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residual);
    res = angle_axis.cwiseQuotient(sigma_q_.template cast<T>());
    return true;
  }

private:
    Eigen::Quaterniond b1_q_b2_;
    Eigen::Vector3d sigma_q_;
};

class PositionPrior {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PositionPrior(const Eigen::Vector3d &p, const Eigen::Vector3d &sigma_p) : p_(p), sigma_p_(sigma_p) {}
    
    template <typename T>
    bool operator()(const T *const p, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_map(p);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residual);
        residual_map = (p_map - p_.template cast<T>()).cwiseQuotient(sigma_p_.template cast<T>());
        return true;
    }

private:
    Eigen::Vector3d p_;
    Eigen::Vector3d sigma_p_;
};


class PositionEdge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PositionEdge(const Eigen::Vector3d &b1_p_b2, const Eigen::Quaterniond &w_q_b1, const Eigen::Vector3d &sigma_p)
        : b1_p_b2_(b1_p_b2), w_q_b1_(w_q_b1), sigma_p_(sigma_p) {}

    template <typename T>
    bool operator()(const T *const p1, const T *const p2, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> w_p_b1(p1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> w_p_b2(p2);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residual);

        residual_map = w_q_b1_.template cast<T>().inverse() * (w_p_b2 - w_p_b1) - b1_p_b2_.template cast<T>();
        residual_map = residual_map.cwiseQuotient(sigma_p_.template cast<T>());

        return true;
    }

private:
    Eigen::Vector3d b1_p_b2_;
    Eigen::Quaterniond w_q_b1_;
    Eigen::Vector3d sigma_p_;
};

class PositionEdge2 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PositionEdge2(const Eigen::Vector3d &E_p_B, const Eigen::Vector3d &L_p_B, const Eigen::Vector3d &sigma_p)
        : E_p_B_(E_p_B), L_p_B_(L_p_B), sigma_p_(sigma_p) {}

    template <typename T>
    bool operator()(const T *const W_T_L, const T *const E_T_W, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> W_p_L(W_T_L);
        Eigen::Map<const Eigen::Quaternion<T>> W_q_L(W_T_L + 3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> E_p_W(E_T_W);
        Eigen::Map<const Eigen::Quaternion<T>> E_q_W(E_T_W + 3);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residual);
        Eigen::Matrix<T, 3, 1> W_p_B = W_p_L + W_q_L * L_p_B_.template cast<T>();
        Eigen::Matrix<T, 3, 1> E_p_B = E_p_W + E_q_W * W_p_B;
        residual_map = (E_p_B - E_p_B_.template cast<T>()).cwiseQuotient(sigma_p_.template cast<T>());
        return true;
    }

private:
    Eigen::Vector3d E_p_B_;
    Eigen::Vector3d L_p_B_;
    Eigen::Vector3d sigma_p_;
};


struct PriorSpeedAndBias {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<double, 9, 1> sbgba;
    Eigen::Matrix<double, 9, 1> sigma;

    PriorSpeedAndBias(const Eigen::Matrix<double, 9, 1> &_sbgba, const Eigen::Matrix<double, 9, 1> &_sigma) : 
        sbgba(_sbgba), sigma(_sigma) {}

    template <typename T>
    bool operator()(const T *const val, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 9, 1>> val_map(val);
        Eigen::Map<Eigen::Matrix<T, 9, 1>> residual_map(residual);
        residual_map = (val_map - sbgba.template cast<T>()).cwiseQuotient(sigma.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double, 9, 1> &sbgba, const Eigen::Matrix<double, 9, 1> &sigma) {
        return new ceres::AutoDiffCostFunction<PriorSpeedAndBias, 9, 9>(new PriorSpeedAndBias(sbgba, sigma));
    }
};

struct PriorRotation2 {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Quaterniond E_q_L_obs;
    Eigen::Vector3d sigma;

    PriorRotation2(const Eigen::Quaterniond &_E_q_L_obs, const Eigen::Vector3d &_sigma) : E_q_L_obs(_E_q_L_obs), sigma(_sigma) {}

    template <typename T>
    bool operator()(const T *const W_T_L, const T *const E_T_W, T *residual) const {
        Eigen::Map<const Eigen::Quaternion<T>> W_q_L(W_T_L + 3);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residual);
        Eigen::Quaternion<T> E_q_W = Eigen::Map<const Eigen::Quaternion<T>>(E_T_W + 3);
        Eigen::Quaternion<T> E_q_L = E_q_W * W_q_L;
        Eigen::Quaternion<T> q_diff = E_q_L.conjugate() * E_q_L_obs.template cast<T>();
        Eigen::AngleAxis<T> aa(q_diff);
        Eigen::Matrix<T, 3, 1> angle_axis = aa.angle() * aa.axis();
        residual_map = angle_axis.cwiseQuotient(sigma.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Quaterniond &E_q_L_obs, const Eigen::Vector3d &sigma) {
        return new ceres::AutoDiffCostFunction<PriorRotation2, 3, 7, 7>(new PriorRotation2(E_q_L_obs, sigma));
    }
};

struct CloseZLoop {
    double sigma;
    CloseZLoop(double _sigma) : sigma(_sigma) {}

    template <typename T>
    bool operator()(const T *const W_T_L1, const T *const W_T_L2, const T *const E_T_W, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> W_p_L1(W_T_L1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> W_p_L2(W_T_L2);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> E_p_W(E_T_W);
        Eigen::Map<const Eigen::Quaternion<T>> E_q_W(E_T_W + 3);
        Eigen::Map<Eigen::Matrix<T, 1, 1>> residual_map(residual);
        Eigen::Matrix<T, 3, 1> E_p_L1 = E_p_W + E_q_W * W_p_L1;
        Eigen::Matrix<T, 3, 1> E_p_L2 = E_p_W + E_q_W * W_p_L2;
        residual_map[0] = (E_p_L1[2] - E_p_L2[2]) / T(sigma);
        return true;
    }

    static ceres::CostFunction *Create(double sigma) {
        return new ceres::AutoDiffCostFunction<CloseZLoop, 1, 7, 7, 7>(new CloseZLoop(sigma));
    }
};
