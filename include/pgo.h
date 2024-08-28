#pragma once
#include <okvis/Time.hpp>
#include <okvis/ImuMeasurements.hpp>

#include <map>

struct StateInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::Time time;
    Eigen::Matrix<double, 7, 1> pose;
    Eigen::Matrix<double, 9, 1> speedAndBias;
    std::shared_ptr<const okvis::ImuMeasurementDeque> imu_til_this;
    bool opt_this_speed = false;

    StateInfo() : time(0.0) {}

    explicit StateInfo(okvis::Time t) : time(t) {}

    StateInfo(okvis::Time t, const Eigen::Matrix<double, 7, 1> &_pose,
              const Eigen::Matrix<double, 9, 1> &_speedAndBias,
              std::shared_ptr<const okvis::ImuMeasurementDeque> _imuSegment)
        : time(t), pose(_pose), speedAndBias(_speedAndBias), imu_til_this(_imuSegment) {}
};


typedef std::map<okvis::Time, StateInfo, std::less<okvis::Time>,
                 Eigen::aligned_allocator<std::pair<const okvis::Time, StateInfo>>>
    StateMap;