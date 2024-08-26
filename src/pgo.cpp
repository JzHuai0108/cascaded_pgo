/**
 * @file pgo.cpp
 * @brief Pose graph optimization
 * Given the absolute pose constraints at the front and the back of a trajectory,
 * and the relative poses between frames from an odometry method,
 * optimize the trajectory to minimize the error in the constraints.
 * This is done in several steps:
 * First only optimize the translations with a ceres problem;
 * Second optimize both rotation and translations with a ceres problem.
 */

#include <pose_factors.h>
#include <ceres/problem.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "okvis/Duration.hpp"
#include "okvis/Time.hpp"
#include "okvis/implementation/Time.hpp"
#include "okvis/implementation/Duration.hpp"
#include <ceres/rotation.h>
#include <RotationManifold.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "pose_factors.h"
#include <gflags/gflags.h>

#include "geodetic_enu_converter.h"

DEFINE_double(cull_begin_secs, 0.0, "Cull the poses at the beginning of the front loc session and the backward loc session to avoid the jittering part");
DEFINE_double(cull_end_secs, 0.0, "Cull the poses at the end of the front loc session and the backward loc session to avoid the drift part");
DEFINE_int32(near_time_tol, 100000, "Tolerance in nanoseconds for finding the nearest pose in time");
DEFINE_double(trans_prior_sigma, 0.1, "Prior sigma for translation");
DEFINE_double(rot_prior_sigma, 0.05, "Prior sigma for rotation");
DEFINE_double(relative_trans_sigma, 0.01, "Relative translation sigma in about 0.1 sec, enlarge this value to reduce small spikes in PGO result.");
DEFINE_double(relative_rot_sigma, 0.005, "Relative rotation sigma in about 0.1 sec, enlarge this value to reduce small spikes in PGO result");
DEFINE_bool(opt_rotation_only, true, "Perform rotation only optimization");
DEFINE_bool(opt_translation_only, true, "Perform translation only optimization");
DEFINE_bool(opt_poses, true, "Perform 6DOF pose graph optimization");
DEFINE_double(gnss_sigma_xy, 0.25, "GNSS position sigma in xy plane");
DEFINE_double(gnss_sigma_z, 1.0, "GNSS position sigma in z");
DEFINE_string(L_p_B, "0 0.06 -0.16", "position of the x36d INS body in the L frame");
DEFINE_string(E_T_tls, "", "pose of the TLS in the GNSS E frame, if provided, it will be locked in PGO.");
DEFINE_double(td, 0.0, "Time delay of the odometry system, td + odometry original times = odometry times in GNSS clock");
DEFINE_double(huber_width, 2.0, "Huber loss width for GNSS position residuals");
DEFINE_bool(use_nhc, true, "Use non-holonomic constraint in PGO");
DEFINE_double(nh_sigma, 0.1, "Non-holonomic constraint sigma for left and up velocity of the vehicle.");
DEFINE_bool(gnss_to_enu, false, "Convert the UTM 50 GNSS positions to ENU frame and then used for PGO constraints");

struct GnssPosition {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::Time time;
    Eigen::Vector3d position;
    Eigen::Vector3d sigma;
    int status; // NavSatFix status

    GnssPosition(const okvis::Time &t, const Eigen::Vector3d &p, const Eigen::Vector3d &s, int st) : time(t), position(p), sigma(s), status(st) {}

    friend std::ostream &operator<<(std::ostream &os, const GnssPosition &gp) {
        os << gp.time << " " << gp.position.transpose() << " " << gp.sigma.transpose() << " " << gp.status;
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

size_t load_poses(const std::string &posefile,
    std::vector<okvis::Time> &times, 
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> &poses,
    okvis::Duration cull_begin_secs = okvis::Duration(0),
    okvis::Duration cull_end_secs = okvis::Duration(0)) {
    std::ifstream stream(posefile);
    if (!stream.is_open()) {
        std::cerr << "Failed to open file " << posefile << std::endl;
        return 1;
    }
    while (!stream.eof()) {
        std::string line;
        std::getline(stream, line);
        if (line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::string time_str;
        iss >> time_str;
        if (time_str.empty()) {
            break;
        }
        // keep the time precision to nanoseconds
        std::string::size_type pos = time_str.find('.');
        uint32_t sec, nsec;
        try {
            sec = std::stoul(time_str.substr(0, pos));
            std::string nsecstr = time_str.substr(pos + 1);
            if (nsecstr.size() < 9) {
                nsecstr.append(9 - nsecstr.size(), '0');
            }
            nsec = std::stoul(nsecstr);
        } catch (std::exception &e) {
            std::cerr << "Failed to parse time string: " << time_str << " in " << posefile << std::endl;
            return 1;
        }
        okvis::Time time(sec, nsec);

        Eigen::Matrix<double, 7, 1> pose;
        for (size_t i = 0; i < 7; ++i) {
            iss >> pose[i];
        }
        times.push_back(time);
        poses.push_back(pose);
    }
    if (times.empty()) {
        std::cerr << "No poses loaded from " << posefile << std::endl;
        return 1;
    }
    // cull the end
    okvis::Time end_time = times.back();
    okvis::Time cull_end_time = end_time - cull_end_secs;
    auto eit = std::lower_bound(times.begin(), times.end(), cull_end_time);
    int m = times.end() - eit;
    times.erase(eit, times.end());
    poses.erase(poses.end() - m, poses.end());

    // cull the begin
    okvis::Time begin_time = times.front();
    okvis::Time cull_begin_time = begin_time + cull_begin_secs;
    auto it = std::lower_bound(times.begin(), times.end(), cull_begin_time);
    int n = it - times.begin();
    times.erase(times.begin(), it);
    poses.erase(poses.begin(), poses.begin() + n);
    return 0;
}

size_t load_gnss_positions(const std::string &gnss_file, 
        PositionVector &gnss_positions, Eigen::Vector3d &anchor_llh, bool gnss_to_enu) {
    std::ifstream stream(gnss_file);
    if (!stream.is_open()) {
        std::cerr << "Failed to open file " << gnss_file << std::endl;
        return 1;
    }
    VecVec3d llh_list;
    llh_list.reserve(1000);
    while (!stream.eof()) {
        std::string line;
        std::getline(stream, line);
        if (line[0] == '#' || line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::string time_str;
        std::getline(iss, time_str, ',');
        // keep the time precision to nanoseconds
        std::string::size_type pos = time_str.find('.');
        uint32_t sec, nsec;
        try {
            sec = std::stoul(time_str.substr(0, pos));
            std::string nsecstr = time_str.substr(pos + 1);
            if (nsecstr.size() < 9) {
                nsecstr.append(9 - nsecstr.size(), '0');
            }
            nsec = std::stoul(nsecstr);
        } catch (std::exception &e) {
            std::cerr << "Failed to parse time string: " << time_str << " in " << gnss_file << std::endl;
            return 1;
        }
        okvis::Time time(sec, nsec);

        Eigen::Vector3d position;
        char delim;
        for (size_t i = 0; i < 3; ++i) {
            iss >> position[i] >> delim;
        }
        Eigen::Matrix<double, 4, 1> quat;
        for (size_t i = 0; i < 4; ++i) {
            iss >> quat[i] >> delim;
        }
        Eigen::Vector3d llh;
        for (size_t i = 0; i < 3; ++i) {
            iss >> llh[i] >> delim;
        }
        llh_list.push_back(llh);
        int status;
        float val;
        for (size_t i = 0;  i < 2; ++i) {
            iss >> val >> delim;
        }
        status = val;

        gnss_positions.emplace_back(time, position, 
                Eigen::Vector3d(FLAGS_gnss_sigma_xy, FLAGS_gnss_sigma_xy, FLAGS_gnss_sigma_z), status);
    }
    stream.close();

    anchor_llh = llh_list.front();
    if (gnss_to_enu) {
        VecVec3d enu_list;
        geodeticToEnu(llh_list, anchor_llh, enu_list);
        for (size_t i = 0; i < enu_list.size(); ++i) {
            gnss_positions[i].position = enu_list[i];
        }
    }

    if (gnss_positions.empty()) {
        std::cerr << "No GNSS positions loaded from " << gnss_file << std::endl;
        return 1;
    }
    std::cout << "Loaded " << gnss_positions.size() << " GNSS positions from " << gnss_file << std::endl;
    std::cout << "First GNSS position at " << gnss_positions.front() << std::endl;
    return 0;
}

void shift_gnss_times(double td, PositionVector &gnss_positions) {
    okvis::Duration d(td);
    for (auto &gp : gnss_positions) {
        gp.time -= d;
    }
}

std::vector<okvis::Time> correct_back_times(std::vector<okvis::Time> &back_times, const okvis::Time &max_bag_time) {
    std::vector<okvis::Time> actual_back_times;
    actual_back_times.reserve(back_times.size());
    for (size_t i = 0; i < back_times.size(); ++i) {
        auto actual_back_time = max_bag_time + (max_bag_time - back_times[i]);
        actual_back_times.push_back(actual_back_time);
    }
    
    return actual_back_times;
}

void load_times(const std::string &timefile, okvis::Time &max_bag_time) {
    // timefile format:
    // first line is max_bag_time,
    // second line is max_bag_time * 2
    std::ifstream stream(timefile);
    if (!stream.is_open()) {
        std::cerr << "Failed to open file " << timefile << std::endl;
        return;
    }
    std::string line;
    std::getline(stream, line);
    std::istringstream iss(line);
    std::string time_str;
    iss >> time_str;
    if (time_str.empty()) {
        return;
    }
    // keep the time precision to nanoseconds
    std::string::size_type pos = time_str.find('.');
    uint32_t sec, nsec;
    try {
        sec = std::stoul(time_str.substr(0, pos));
        nsec = std::stoul(time_str.substr(pos + 1));
    } catch (std::exception &e) {
        std::cerr << "Failed to parse time string: " << time_str << " in " << timefile << std::endl;
        return;
    }
    max_bag_time = okvis::Time(sec, nsec);
}


void outputOptimizedPoses(const std::map<okvis::Time, Eigen::Matrix<double, 7, 1>, std::less<okvis::Time>, Eigen::aligned_allocator<std::pair<const okvis::Time, Eigen::Matrix<double, 7, 1>>>>& optimized_poses, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (const auto& pose : optimized_poses) {
        const Eigen::Matrix<double, 7, 1>& T_ab = pose.second;

        Eigen::Vector3d position = T_ab.block<3, 1>(0, 0);
        Eigen::Quaterniond orientation(T_ab.block<4, 1>(3, 0));

        file << pose.first.sec << "." << std::setw(9) << std::setfill('0') << pose.first.nsec << " "
             << std::setprecision(9) << std::fixed << position.x() << " " << position.y() << " " << position.z() << " "
             << orientation.x() << " " << orientation.y() << " " << orientation.z() << " " << orientation.w() << std::endl;
    }

    file.close();
}

okvis::Time findNearestTime(
    const std::map<okvis::Time, Eigen::Matrix<double, 7, 1>,
                   std::less<okvis::Time>,
                   Eigen::aligned_allocator<std::pair<const okvis::Time, Eigen::Matrix<double, 7, 1>>>>& optimized_poses,
    const okvis::Time& time, int tol_ns) {
    // find the nearest time in optimized_poses to the given time
    std::pair<okvis::Time, Eigen::Matrix<double, 7, 1>> target_pose(time, Eigen::Matrix<double, 7, 1>::Zero());
    auto uit = std::upper_bound(optimized_poses.begin(), optimized_poses.end(), target_pose,
                                [](const std::pair<okvis::Time, Eigen::Matrix<double, 7, 1>>& lhs,
                                   const std::pair<okvis::Time, Eigen::Matrix<double, 7, 1>>& rhs) {
                                    return lhs.first < rhs.first;
                                });

    if (uit == optimized_poses.end()) {
        auto it = std::prev(uit);
        okvis::Duration d = time - it->first;
        if (d < okvis::Duration(0, tol_ns)) {
            return it->first;
        } else {
            std::cerr << "At tail, no nearest time found for " << time.sec << "." << std::setw(9) << std::setfill('0') << time.nsec << std::endl;
            return okvis::Time();
        }
    } else {
        auto it = uit;
        if (it == optimized_poses.begin()) {
            okvis::Duration d = it->first - time;
            if (d < okvis::Duration(0, tol_ns)) {
                return it->first;
            } else {
                std::cerr << "At head, no nearest time found for " << time.sec << "." << std::setw(9) << std::setfill('0') << time.nsec << std::endl;
                return okvis::Time();
            }
        } else {
            auto it_prev = std::prev(it);
            okvis::Duration d1 = it->first - time;
            okvis::Duration d2 = time - it_prev->first;
            if (d1 < d2) {
                if (d1 < okvis::Duration(0, tol_ns)) {
                    return it->first;
                } else {
                    std::cerr << "No nearest time found for " << time.sec << "." << std::setw(9) << std::setfill('0') << time.nsec
                         << ". The nearest duration is " << d1.toSec() << " s" << std::endl;
                    return okvis::Time();
                }
            } else {
                if (d2 < okvis::Duration(0, tol_ns)) {
                    return it_prev->first;
                } else {
                    std::cerr << "No nearest time found for " << time.sec << "." << std::setw(9) << std::setfill('0') << time.nsec
                            << ". The nearest duration is " << d2.toSec() << " s" << std::endl;
                    return okvis::Time();
                }
            }
        }
    }
}

// check whether a file exists
bool exists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void associateAndUpdate(const std::vector<okvis::Time> &odom_times,
    const std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> &odom_poses,
    std::vector<okvis::Time> &loc_times,
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> &loc_poses) {
    std::vector<okvis::Time> new_loc_times;
    new_loc_times.reserve(loc_times.size());
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> new_loc_poses;
    new_loc_poses.reserve(loc_poses.size());

    for (int i = 0; i < (int)loc_times.size(); ++i) {
        auto it = std::upper_bound(odom_times.begin(), odom_times.end(), loc_times[i]);
        if (it == odom_times.end()) {
            std::cerr << "No upper odometry time found for " << loc_times[i] << ", max odometry time is " << odom_times.back() << std::endl;
            break;
        }
        okvis::Time odomtime = *it;
        okvis::Time left = loc_times[i];
        int leftid = i;
        int rightid = i + 1;
        while (rightid < (int)loc_times.size() && loc_times[rightid] < odomtime) {
            ++rightid;
        }
        if (rightid == (int)loc_times.size()) {
            std::cerr << "No right loc time found for " << odomtime << ", max loc time is " << loc_times.back() << std::endl;
            break;
        }
        okvis::Time right = loc_times[rightid];
        double ratio = (odomtime - left).toSec() / (right - left).toSec();
        Eigen::Matrix<double, 7, 1> new_loc_pose;
        new_loc_pose.head<3>() = loc_poses[leftid].head<3>() + ratio * (loc_poses[rightid].head<3>() - loc_poses[leftid].head<3>());
        Eigen::Quaterniond quat_left(loc_poses[leftid](6), loc_poses[leftid](3), loc_poses[leftid](4), loc_poses[leftid](5));
        Eigen::Quaterniond quat_right(loc_poses[rightid](6), loc_poses[rightid](3), loc_poses[rightid](4), loc_poses[rightid](5));
        Eigen::Quaterniond quat_new = quat_left.slerp(ratio, quat_right);
        new_loc_pose.block<4, 1>(3, 0) = quat_new.coeffs();
        new_loc_times.push_back(odomtime);
        new_loc_poses.push_back(new_loc_pose);
    }
    loc_times = new_loc_times;
    loc_poses = new_loc_poses;

}

class CascadedPgo {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    std::vector<okvis::Time> tls_times, tls_times_orig;
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> tls_poses, tls_poses_orig;
 
    std::vector<okvis::Time> odometry_times;
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> odometry_poses;

    std::map<okvis::Time, Eigen::Matrix<double, 7, 1>, std::less<okvis::Time>, 
        Eigen::aligned_allocator<std::pair<const okvis::Time, Eigen::Matrix<double, 7, 1>>>> optimized_poses;

    PositionVector gnss_positions;
    Eigen::Vector3d anchor_llh;
    Eigen::Matrix4d init_E_T_tls;
    Eigen::Matrix<double, 7, 1> est_E_T_tls;
    Eigen::Vector3d L_p_B_;

    CascadedPgo(const std::string &odometry_file, const std::string &tls_loc_file, const std::string &gnss_loc_file) {
        okvis::Duration cull_end_secs(FLAGS_cull_end_secs);
        okvis::Duration cull_begin_secs(FLAGS_cull_begin_secs);
        load_poses(tls_loc_file, tls_times, tls_poses, cull_begin_secs, cull_end_secs);

        load_poses(odometry_file, odometry_times, odometry_poses);

        if (!gnss_loc_file.empty()) {
            load_gnss_positions(gnss_loc_file, gnss_positions, anchor_llh, FLAGS_gnss_to_enu);
            shift_gnss_times(FLAGS_td, gnss_positions);
            L_p_B_ = Eigen::Matrix<double, 3, 1>::Zero();
            std::istringstream iss(FLAGS_L_p_B);
            for (int i = 0; i < 3; ++i) {
                iss >> L_p_B_[i];
            }
            std::cout << "L_p_B: " << L_p_B_.transpose() << std::endl;
        }
    }

    bool useGnss() const {
        return !gnss_positions.empty();
    }

    bool useEnuGnss() const {
        return !gnss_positions.empty() && FLAGS_gnss_to_enu;
    }

    void associateAndInterpolatePoses() {
        tls_times_orig = tls_times;
        tls_poses_orig = tls_poses;
        associateAndUpdate(odometry_times, odometry_poses, tls_times, tls_poses);
        std::cout << "Associated new poses " << tls_times.size() << " from "
            << tls_times.front() << " to " << tls_times.back() << " of duration "
            << (tls_times.back() - tls_times.front()).toSec() << " s" << std::endl;
    }

    void InitializePoses() {
        // Compute transformation from odometry to front using the first poses of front and odometry
        Eigen::Quaterniond w_rot_wodom;
        Eigen::Vector3d w_trans_wodom;

        Eigen::Matrix<double, 3, -1> odom_points(3, tls_times.size());
        Eigen::Matrix<double, 3, -1> front_points(3, tls_times.size());
        for (size_t i = 0; i < tls_times.size(); ++i) {
            auto it = std::lower_bound(odometry_times.begin(), odometry_times.end(), tls_times[i]);
            size_t j = it - odometry_times.begin();    
            Eigen::Quaterniond quat_front(tls_poses[i](6), tls_poses[i](3), tls_poses[i](4), tls_poses[i](5));
            Eigen::Quaterniond quat_odom(odometry_poses[j](6), odometry_poses[j](3), odometry_poses[j](4), odometry_poses[j](5));
            Eigen::Vector3d trans_front(tls_poses[i].block<3, 1>(0, 0));
            Eigen::Vector3d trans_odom(odometry_poses[j].block<3, 1>(0, 0));
            Eigen::Matrix4d T_front = Eigen::Matrix4d::Identity();
            T_front.block<3, 3>(0, 0) = quat_front.toRotationMatrix();
            T_front.block<3, 1>(0, 3) = trans_front;
            Eigen::Matrix4d L_T_odom = Eigen::Matrix4d::Identity();
            L_T_odom.block<3, 3>(0, 0) = quat_odom.conjugate().toRotationMatrix();
            L_T_odom.block<3, 1>(0, 3) = quat_odom.conjugate() * (-trans_odom);
            Eigen::Matrix4d w_T_wodom = T_front * L_T_odom;
            w_rot_wodom = Eigen::Quaterniond(w_T_wodom.block<3, 3>(0, 0));
            w_trans_wodom = w_T_wodom.block<3, 1>(0, 3);
            std::cout << "w_trans_wodom: " << w_trans_wodom.transpose() << std::endl;
            std::cout << "w_rot_wodom: " << w_rot_wodom.toRotationMatrix() << std::endl;
            break;
        }

        // Refined by Umeyama method, but this is unstable.
        // for (size_t i = 0; i < tls_times.size(); ++i) {
        //     auto it = std::lower_bound(odometry_times.begin(), odometry_times.end(), tls_times[i]);
        //     if (it == odometry_times.end()) {
        //         continue;
        //     }
        //     if (*it - tls_times[i] > okvis::Duration(0.001)) {
        //         std::cout << "Front time " << tls_times[i] << " is not close to odometry time " << *it << std::endl;
        //     }
        //     odom_points.col(i) = odometry_poses[it - odometry_times.begin()].block<3, 1>(0, 0);
        //     front_points.col(i) = tls_poses[i].block<3, 1>(0, 0);
        // }
        // Eigen::Matrix4d w_T_wodom = Eigen::umeyama(odom_points, front_points, false);
        // std::cout << "Initial W_T_odom by Umeyama: " << std::endl << w_T_wodom << std::endl;
        // w_rot_wodom = Eigen::Quaterniond(w_T_wodom.block<3, 3>(0, 0));
        // w_trans_wodom = w_T_wodom.block<3, 1>(0, 3);

        for (size_t i = 0; i < odometry_times.size(); ++i) {
            Eigen::Matrix<double, 7, 1> pose;
            Eigen::Quaterniond quat(odometry_poses[i](6), odometry_poses[i](3), odometry_poses[i](4), odometry_poses[i](5));
            quat = w_rot_wodom * quat;
            Eigen::Vector3d trans = w_rot_wodom * odometry_poses[i].block<3, 1>(0, 0) + w_trans_wodom;
            pose.block<3, 1>(0, 0) = trans;
            pose.block<4, 1>(3, 0) = quat.coeffs();
            optimized_poses[odometry_times[i]] = pose;
        }
        // add the front and back poses not in odometry_times
        size_t i = 0;
        size_t n = 0;
        for (; i < tls_times_orig.size(); ++i) {
            okvis::Time t = tls_times_orig[i];
            if (odometry_times[0] - t > okvis::Duration(0.05)) {
                optimized_poses[t] = tls_poses_orig[i];
                ++n;
            } else {
                break;
            }
        }
        if (n) {
            std::cout << "tls_times front " << tls_times.front() << ", orig front " << tls_times_orig.front() 
                << ", orig cut " << tls_times_orig[i-1] << ", n " << n << std::endl;

            tls_times.insert(tls_times.begin(), tls_times_orig.begin(), tls_times_orig.begin() + i);
            tls_poses.insert(tls_poses.begin(), tls_poses_orig.begin(), tls_poses_orig.begin() + i);
        }

        size_t s = 0;
        n = 0;
        for (i = 0; i < tls_times_orig.size(); ++i) {
            okvis::Time t = tls_times_orig[i];
            if (t - odometry_times.back() > okvis::Duration(0.05)) {
                optimized_poses[t] = tls_poses_orig[i];
                if (s == 0) {
                    s = i;
                }
                ++n;
            }
        }
        if (n) {
            std::cout << "tls_times back " << tls_times.back() << ", orig cut " 
                    << tls_times_orig[s] << ", orig back " << tls_times_orig.back() << ", n " << n << std::endl;
            tls_times.insert(tls_times.end(), tls_times_orig.begin() + s, tls_times_orig.begin() + i);
            tls_poses.insert(tls_poses.end(), tls_poses_orig.begin() + s, tls_poses_orig.begin() + i);
        }
    }

    void FitGnssPositions() {
        if (gnss_positions.size()) {
            // interpolate the gnss positions at optimized pose times, also compute the robust E_T_W.
            PositionVector fitted_gnss_positions;
            int validindex = 0;
            Eigen::Matrix<double, 3, -1> tls_points(3, optimized_poses.size());
            Eigen::Matrix<double, 3, -1> gnss_points(3, optimized_poses.size());
            for (const auto& tp : optimized_poses) {
                okvis::Time t = tp.first;
                GnssPosition p(t, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0);
                auto it = std::upper_bound(gnss_positions.begin(), gnss_positions.end(), p);
                if (it == gnss_positions.end() || it == gnss_positions.begin()) {
                    continue;
                }
                auto it_prev = std::prev(it);
                okvis::Duration dt = it->time - it_prev->time;
                if (dt > okvis::Duration(1.0)) {
                    continue;
                }
                double ratio = (t - it_prev->time).toSec() / (it->time - it_prev->time).toSec();
                Eigen::Vector3d pos = it_prev->position + ratio * (it->position - it_prev->position);
                fitted_gnss_positions.emplace_back(t, pos, it->sigma, it->status);

                tls_points.col(validindex) = tp.second.block<3, 1>(0, 0);
                gnss_points.col(validindex) = pos;
                ++validindex;
            }
            std::cout << "Fitted " << fitted_gnss_positions.size() << " GNSS positions from " << gnss_positions.size() << std::endl;
            gnss_positions = fitted_gnss_positions;

            tls_points.conservativeResize(3, validindex);
            gnss_points.conservativeResize(3, validindex);

            if (FLAGS_E_T_tls.empty() || FLAGS_gnss_to_enu) {
                init_E_T_tls = Eigen::umeyama(tls_points, gnss_points, false);
                std::cout << "Initial E_T_W by Umeyama: " << std::endl << init_E_T_tls << std::endl;
                est_E_T_tls = Eigen::Matrix<double, 7, 1>::Zero();
                est_E_T_tls.head<3>() = init_E_T_tls.block<3, 1>(0, 3);
                Eigen::Quaterniond quat(init_E_T_tls.block<3, 3>(0, 0));
                est_E_T_tls.block<4, 1>(3, 0) = quat.coeffs();
            } else {
                est_E_T_tls = Eigen::Matrix<double, 7, 1>::Zero();
                std::istringstream iss(FLAGS_E_T_tls);
                for (int i = 0; i < 7; ++i) {
                    iss >> est_E_T_tls[i];
                }

                init_E_T_tls = Eigen::Matrix4d::Identity();
                init_E_T_tls.block<3, 3>(0, 0) = Eigen::Quaterniond(est_E_T_tls.block<4, 1>(3, 0)).toRotationMatrix();
                init_E_T_tls.block<3, 1>(0, 3) = est_E_T_tls.block<3, 1>(0, 0);
                std::cout << "Initial E_T_W from FLAGS: " << std::endl << init_E_T_tls << std::endl;
            }
        }
    }

    void OptimizeRotation() {
        std::cout << "Optimizing rotation..." << std::endl;
        ceres::Problem rotation_optimizer;
        ceres::Manifold *rotmanifold = new ceres::RotationManifold();
        for (auto &pose : optimized_poses) {
            rotation_optimizer.AddParameterBlock(pose.second.data() + 3, 4, rotmanifold);
        }

        int n = 0;
        for (size_t i = 0; i < tls_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, tls_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                std::cerr << "TLS pose not found in optimized poses!" << std::endl;
                continue;
            }
            Eigen::Quaterniond quat(tls_poses[i](6), tls_poses[i](3), tls_poses[i](4), tls_poses[i](5));      
            SO3Prior* so3prior = new SO3Prior(quat, Eigen::Vector3d(FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma));
            rotation_optimizer.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SO3Prior, 3, 4>(so3prior),
                nullptr, optimized_poses[nearest_time].data() + 3);
            ++n;
        }
        std::cout << "Added " << n << " TLS prior rotations." << std::endl;

        // add odometry constraints
        n = 0;
        for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
            okvis::Time ot = odometry_times[i];
            Eigen::Quaterniond quat(odometry_poses[i](6), odometry_poses[i](3), odometry_poses[i](4), odometry_poses[i](5));
            Eigen::Quaterniond quat_next(odometry_poses[i + 1](6), odometry_poses[i + 1](3), odometry_poses[i + 1](4), odometry_poses[i + 1](5));
            Eigen::Quaterniond b1_q_b2 = quat.inverse() * quat_next;

            SO3Edge* so3edge = new SO3Edge(b1_q_b2, Eigen::Vector3d(FLAGS_relative_rot_sigma, FLAGS_relative_rot_sigma, FLAGS_relative_rot_sigma));
            rotation_optimizer.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SO3Edge, 3, 4, 4>(so3edge),
                nullptr, optimized_poses[odometry_times[i]].data() + 3, optimized_poses[odometry_times[i + 1]].data() + 3);
            ++n;
        }
        std::cout << "Added " << n << " relative odometry constraints." << std::endl;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 20;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &rotation_optimizer, &summary);
    }

    void OptimizeTranslation() {
        std::cout << "Optimizing translation..." << std::endl;
        ceres::Problem translation_optimizer;
        // add prior translations from the front and the back
        for (size_t i = 0; i < tls_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, tls_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                std::cerr << "TLS pose not found in optimized poses!" << std::endl;
                continue;
            }
            PositionPrior* front_prior = new PositionPrior(tls_poses[i].block<3, 1>(0, 0),
                    Eigen::Vector3d(FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma));
            translation_optimizer.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PositionPrior, 3, 3>(front_prior), 
                nullptr, optimized_poses[nearest_time].data());
        }

        for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
            okvis::Time ot = odometry_times[i];
            Eigen::Vector3d w_p_b1b2 = odometry_poses[i + 1].block<3, 1>(0, 0) - odometry_poses[i].block<3, 1>(0, 0);
            Eigen::Quaterniond b1_q(odometry_poses[i](6), odometry_poses[i](3), odometry_poses[i](4), odometry_poses[i](5));
            Eigen::Vector3d b1_p_b2 = b1_q.inverse() * w_p_b1b2;

            Eigen::Quaterniond w_q_b1_new(optimized_poses[odometry_times[i]].block<4, 1>(3, 0));
            PositionEdge* edge = new PositionEdge(b1_p_b2, w_q_b1_new, 
                    Eigen::Vector3d(FLAGS_relative_trans_sigma, FLAGS_relative_trans_sigma, FLAGS_relative_trans_sigma));
            translation_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<PositionEdge, 3, 3, 3>(edge),
                nullptr, optimized_poses[odometry_times[i]].data(), optimized_poses[odometry_times[i + 1]].data());
            Eigen::Vector3d error;
            edge->operator()(optimized_poses[odometry_times[i]].data(), optimized_poses[odometry_times[i + 1]].data(), error.data());
            if (error.norm() > 1 / FLAGS_relative_trans_sigma) {
                Eigen::Vector3d w_p_b1 = optimized_poses[odometry_times[i]].block<3, 1>(0, 0);
                Eigen::Vector3d w_p_b2 = optimized_poses[odometry_times[i + 1]].block<3, 1>(0, 0);
                Eigen::Vector3d pred = w_q_b1_new.inverse() * (w_p_b2 - w_p_b1);
                error = pred - b1_p_b2;
                std::cout << "Predicted b1_p_b2: " << pred.x() << " " << pred.y() << " " << pred.z() << std::endl;
                std::cout << "Raw error: " << error.x() << " " << error.y() << " " << error.z() << std::endl;
                std::cerr << "Large error in odometry edge: " << error.norm() << std::endl;
            }
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 20;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &translation_optimizer, &summary);
    }

    void OptimizePoseGraph(const std::string &output_path) {
        std::cout << "Optimizing 6DoF pose graph..." << std::endl;
        ceres::Problem pose_graph_optimizer;
        ceres::Manifold *pose_manifold = new ceres::PoseManifold();
        ceres::LossFunction *loss_function = new ceres::HuberLoss(FLAGS_huber_width);

        for (auto &pose : optimized_poses) {
            pose_graph_optimizer.AddParameterBlock(pose.second.data(), 7, pose_manifold); 
        }
        if (gnss_positions.size() > 0) {
            pose_graph_optimizer.AddParameterBlock(est_E_T_tls.data(), 7, pose_manifold);
            if (!FLAGS_E_T_tls.empty() && !FLAGS_gnss_to_enu) {
                pose_graph_optimizer.SetParameterBlockConstant(est_E_T_tls.data());
            }
        }

        for (size_t i = 0; i < tls_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, tls_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                std::cerr << "TLS pose not found in optimized poses!" << std::endl;
                continue;
            }
            Eigen::Matrix<double, 6, 1> sigmaPQ;
            sigmaPQ << FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma,
                    FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma;
            ceres::CostFunction *prior = EdgeSE3Prior::Create(tls_poses[i], sigmaPQ);
            pose_graph_optimizer.AddResidualBlock(prior, nullptr, optimized_poses[nearest_time].data());
        }

        for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
            okvis::Time ot = odometry_times[i];
            okvis::Time ot_next = odometry_times[i + 1];
            double dt = (ot_next - ot).toSec();
            Eigen::Vector3d w_p_b1b2 = odometry_poses[i + 1].block<3, 1>(0, 0) - odometry_poses[i].block<3, 1>(0, 0);
            Eigen::Quaterniond q_b1(odometry_poses[i].block<4, 1>(3, 0));
            Eigen::Vector3d b1_p_b2 = q_b1.inverse() * w_p_b1b2;

            Eigen::Quaterniond q_b2(odometry_poses[i + 1].block<4, 1>(3, 0));
            Eigen::Quaterniond q_relative = q_b1.inverse() * q_b2;

            Eigen::Matrix<double, 7, 1> T_ab;
            T_ab.block<3, 1>(0, 0) = b1_p_b2;
            T_ab.block<4, 1>(3, 0) = q_relative.coeffs();
            Eigen::Matrix<double, 6, 1> sigmaPQ;
            sigmaPQ << FLAGS_relative_trans_sigma, FLAGS_relative_trans_sigma, FLAGS_relative_trans_sigma,
                    FLAGS_relative_rot_sigma, FLAGS_relative_rot_sigma, FLAGS_relative_rot_sigma;
            ceres::CostFunction* edge = EdgeSE3::Create(T_ab, sigmaPQ);
            pose_graph_optimizer.AddResidualBlock(edge, nullptr, optimized_poses[odometry_times[i]].data(),
                    optimized_poses[odometry_times[i + 1]].data());
            if (FLAGS_use_nhc) {
                ceres::CostFunction* nhc = NonHolonimicConstraint::Create(
                    dt, Eigen::Vector2d(FLAGS_nh_sigma, FLAGS_nh_sigma));
                pose_graph_optimizer.AddResidualBlock(nhc, nullptr, optimized_poses[odometry_times[i]].data(),
                        optimized_poses[odometry_times[i + 1]].data());
            }
        }

        FitGnssPositions();
        std::vector<okvis::Duration> deltas;
        deltas.reserve(gnss_positions.size());
        size_t count = 0;
        if (gnss_positions.size()) {
            okvis::Time last_time;
            int step = 5;
            std::string gnss_error_file = output_path + "/gnss_errors.txt";
            std::cout << "Writing GNSS errors to " << gnss_error_file << std::endl;
            std::ofstream ofs(gnss_error_file, std::ios::out);
            for (size_t i = 0; i < gnss_positions.size(); i += step) {
                okvis::Time t = gnss_positions[i].time;
                int status = gnss_positions[i].status;
                auto nearest_time = findNearestTime(optimized_poses, t, FLAGS_near_time_tol);
                if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                    std::cerr << "GNSS pose not found in optimized poses!" << std::endl;
                    continue;
                }
                Eigen::Vector3d sigma(FLAGS_gnss_sigma_xy, FLAGS_gnss_sigma_xy, FLAGS_gnss_sigma_z);
                if (status != 0) {
                    sigma *= 10;
                }
                PositionEdge2 *edge = new PositionEdge2(gnss_positions[i].position, L_p_B_, sigma);
                auto costfunction = new ceres::AutoDiffCostFunction<PositionEdge2, 3, 7, 7>(edge);
                pose_graph_optimizer.AddResidualBlock(costfunction,
                        loss_function, optimized_poses[nearest_time].data(), est_E_T_tls.data());
                Eigen::Vector3d error;
                edge->operator()(optimized_poses[nearest_time].data(), est_E_T_tls.data(), error.data());
                ofs << t << " " << error[0] << " " << error[1] << " " << error[2] << " " << error.norm() << std::endl;
                if (count > 0) {
                    okvis::Duration delta = t - last_time;
                    deltas.push_back(delta);
                }
                ++count;
                last_time = t;
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &pose_graph_optimizer, &summary);
        if (deltas.size() > 0) {
            std::stringstream ss;
            ss << "Added " << count << " GNSS constraints out of " << gnss_positions.size();
            std::sort(deltas.begin(), deltas.end());
            ss << ", min gap " << deltas.front().toSec() << " s, max gap " << deltas.back().toSec() << " s";
            ss << ", median gap " << deltas[deltas.size() / 2].toSec() << " s" << std::endl;

            ss << "Initial E_T_tls: " << std::endl;
            ss << std::fixed << std::setprecision(9);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ss << init_E_T_tls(i, j) << " ";
                }
                ss << std::endl;
            }
            ss << std::endl;
            Eigen::Quaterniond quat(est_E_T_tls.block<4, 1>(3, 0));
            Eigen::Matrix4d est_E_T_tls_mat = Eigen::Matrix4d::Identity();
            est_E_T_tls_mat.block<3, 3>(0, 0) = quat.toRotationMatrix();
            est_E_T_tls_mat.block<3, 1>(0, 3) = est_E_T_tls.block<3, 1>(0, 0);
            ss << "Estimated E_T_tls: " << std::endl;
            ss << std::fixed << std::setprecision(9);
            for (int i = 0; i < 7; ++i) {
                ss << est_E_T_tls[i] << " ";
            }
            ss << std::endl;
            ss << "Estimated E_T_tls matrix: " << std::endl;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ss << est_E_T_tls_mat(i, j) << " ";
                }
                ss << std::endl;
            }
            ss << std::endl;

            std::ofstream ofs(output_path + "/gnss_result.txt", std::ios::out);
            ofs << ss.str();
            ofs.close();
            std::cout << ss.str();
        }
    }

    void saveResults(const std::string &output_file) const {
        outputOptimizedPoses(optimized_poses, output_file); 
    }

    void saveGeorefResults(const std::string &output_file) const {
        std::ofstream ofs(output_file, std::ios::out);
        ofs << std::fixed << std::setprecision(9);
        for (const auto &pose : optimized_poses) {
            Eigen::Quaterniond W_q_L(pose.second.block<4, 1>(3, 0));
            Eigen::Vector3d W_p_L = pose.second.block<3, 1>(0, 0);
            Eigen::Quaterniond E_q_W(est_E_T_tls.block<4, 1>(3, 0));
            Eigen::Vector3d E_p_W = est_E_T_tls.block<3, 1>(0, 0);
            Eigen::Quaterniond E_q_L = E_q_W * W_q_L;
            Eigen::Vector3d E_p_L = E_p_W + E_q_W * W_p_L;
            ofs << pose.first << " " << E_p_L.x() << " " << E_p_L.y() << " " << E_p_L.z() << " "
                << E_q_L.x() << " " << E_q_L.y() << " " << E_q_L.z() << " " << E_q_L.w() << std::endl;
        }
        ofs.close();
    }

    void saveEnuPoses(const std::string &enuFile) const {
        std::ofstream file(enuFile, std::ios::out);
        file << std::fixed << std::setprecision(9);
        for (size_t i = 0; i < gnss_positions.size(); ++i) {
            Eigen::Vector3d enu = gnss_positions[i].position;
            file << gnss_positions[i].time << " " << enu.x() << " " << enu.y() << " " << enu.z() << std::endl;
        }
        file.close();
        std::cout << "ENU coordinates saved to " << enuFile << std::endl;
    }

}; // class CascadedPgo


int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " odom_file tls_loc_file gnss_loc_file output_path [and other gflags]" << std::endl;
        std::cout << "odom_file contains the poses of L frame in the odometry frame" << std::endl;
        std::cout << "tls_loc_file contains the poses of L frame in the world frame" << std::endl;
        std::cout << "GNSS position file for GNSS constraints E_p_B, B is the IMU frame of the GNSS/INS system." << std::endl;
        std::cout << "output_path is the folder to save the optimized poses W_T_L" << std::endl;
        return 1;
    }

    google::ParseCommandLineFlags(&argc, &argv, true);
    std::string odometry_file = argv[1];
    std::string tls_loc_file = argv[2];
    std::string gnss_loc_file = argv[3];
    std::string output_path = argv[4];
    std::cout << "Odometry: " << odometry_file << std::endl;
    std::cout << "tls loc: " << tls_loc_file << std::endl;
    std::cout << "GNSS loc: " << gnss_loc_file << std::endl;
    std::cout << "Output path: " << output_path << std::endl;
    std::cout << "Cull begin seconds: " << FLAGS_cull_begin_secs << std::endl;
    std::cout << "Cull end seconds: " << FLAGS_cull_end_secs << std::endl;
    CascadedPgo cpgo(odometry_file, tls_loc_file, gnss_loc_file);
    cpgo.associateAndInterpolatePoses();
    cpgo.InitializePoses();
    cpgo.saveResults(output_path + "/initial_poses.txt");
    if (cpgo.useEnuGnss()) {
        std::string enu_file = output_path + "/enu_poses.txt";
        cpgo.saveEnuPoses(enu_file);
    }
    if (FLAGS_opt_rotation_only) {
        cpgo.OptimizeRotation();
        cpgo.saveResults(output_path + "/rotated_poses.txt");
    }
    if (FLAGS_opt_translation_only) {
        cpgo.OptimizeTranslation();
        cpgo.saveResults(output_path + "/translated_poses.txt");
    }
    if (FLAGS_opt_poses) {
        cpgo.OptimizePoseGraph(output_path);
        std::string final_poses_file = output_path + "/final_poses.txt";
        cpgo.saveResults(final_poses_file);
        if (cpgo.useGnss()) {
            final_poses_file = output_path + "/final_georef_poses.txt";
            cpgo.saveGeorefResults(final_poses_file);
        }
    }
}
