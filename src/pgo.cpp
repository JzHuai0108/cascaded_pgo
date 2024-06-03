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

DEFINE_int32(near_time_tol, 100000, "Tolerance in nanoseconds for finding the nearest pose in time");
DEFINE_double(trans_prior_sigma, 0.1, "Prior sigma for translation");
DEFINE_double(rot_prior_sigma, 0.05, "Prior sigma for rotation");
DEFINE_double(relative_trans_sigma, 0.01, "Relative translation sigma in about 0.1 sec");
DEFINE_double(relative_rot_sigma, 0.005, "Relative rotation sigma in about 0.1 sec");

class SO3Prior {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SO3Prior(const Eigen::Quaterniond &q, const Eigen::Vector3d &sigma_q) : q_(q), sigma_q_(sigma_q) {
  }

  template <typename T>
  bool operator()(const T *const q, T *residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> q_map(q);

    Eigen::Quaternion<T> q_conj = q_.cast<T>().inverse();
    Eigen::Quaternion<T> q_diff = q_map * q_conj;

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

class SO3Edge {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SO3Edge(const Eigen::Quaterniond &b1_q_b2, const Eigen::Vector3d &sigma_q) : b1_q_b2_(b1_q_b2), sigma_q_(sigma_q) {}

  template <typename T>
  bool operator()(const T *const q1, const T *const q2, T *residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> q1_map(q1);
    Eigen::Map<const Eigen::Quaternion<T>> q2_map(q2);

    Eigen::Quaternion<T> q1_inv = q1_map.conjugate();
    Eigen::Quaternion<T> q_diff = q1_inv * q2_map * b1_q_b2_.template cast<T>();

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

size_t load_poses(const std::string &posefile,
    std::vector<okvis::Time> &times, 
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> &poses,
    okvis::Duration cull_end_secs = okvis::Duration(0)) {
    // load poses from file
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

    okvis::Time end_time = times.back();
    okvis::Time cull_end_time = end_time - cull_end_secs;
    while (!times.empty() && times.back() > cull_end_time) {
        times.pop_back();
        poses.pop_back();
    }

    return 0;
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

class CascadedPgo {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    std::vector<okvis::Time> front_times;
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> front_poses;

    std::vector<okvis::Time> back_times;
    std::vector<okvis::Time> actual_back_times;
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> back_poses;
 
    std::vector<okvis::Time> odometry_times;
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> odometry_poses;

    std::map<okvis::Time, Eigen::Matrix<double, 7, 1>, std::less<okvis::Time>, 
        Eigen::aligned_allocator<std::pair<const okvis::Time, Eigen::Matrix<double, 7, 1>>>> optimized_poses;

    CascadedPgo(const std::string &front_loc_file, const std::string &back_loc_file, const std::string &odometry_file,
            const std::string &back_time_file) {
        okvis::Duration cull_end_secs(8);
        load_poses(front_loc_file, front_times, front_poses, cull_end_secs);
        load_poses(back_loc_file, back_times, back_poses, cull_end_secs);

        load_poses(odometry_file, odometry_times, odometry_poses);
        okvis::Time max_bag_time;
        load_times(back_time_file, max_bag_time);
        actual_back_times = correct_back_times(back_times, max_bag_time);
    }

    void InitializePoses() {
        // Compute transformation from odometry to front using the first poses of front and odometry
        Eigen::Quaterniond w_rot_wodom;
        Eigen::Vector3d w_trans_wodom;
        bool transformation_found = false;
        for (size_t i = 0; i < front_times.size() && !transformation_found; ++i) {
            if (!odometry_times.empty()) {
                auto it = std::lower_bound(odometry_times.begin(), odometry_times.end(), front_times[i]);
                if (it == odometry_times.end()) {
                    continue;
                }
                size_t j = it - odometry_times.begin();    
                Eigen::Quaterniond quat_front(front_poses[i](6), front_poses[i](3), front_poses[i](4), front_poses[i](5));
                Eigen::Quaterniond quat_odom(odometry_poses[j](6), odometry_poses[j](3), odometry_poses[j](4), odometry_poses[j](5));
                Eigen::Vector3d trans_front(front_poses[i].block<3, 1>(0, 0));
                Eigen::Vector3d trans_odom(odometry_poses[j].block<3, 1>(0, 0));
                Eigen::Matrix4d T_front = Eigen::Matrix4d::Identity();
                T_front.block<3, 3>(0, 0) = quat_front.toRotationMatrix();
                T_front.block<3, 1>(0, 3) = trans_front;
                Eigen::Matrix4d T_odom = Eigen::Matrix4d::Identity();
                T_odom.block<3, 3>(0, 0) = quat_odom.toRotationMatrix();
                T_odom.block<3, 1>(0, 3) = trans_odom;
                Eigen::Matrix4d w_T_wodom = T_front * T_odom.inverse();
                w_rot_wodom = Eigen::Quaterniond(w_T_wodom.block<3, 3>(0, 0));
                w_trans_wodom = w_T_wodom.block<3, 1>(0, 3);
                transformation_found = true;
            }
        }
        
        for (size_t i = 0; i < odometry_times.size(); ++i) {
            Eigen::Matrix<double, 7, 1> pose;
            Eigen::Quaterniond quat(odometry_poses[i](6), odometry_poses[i](3), odometry_poses[i](4), odometry_poses[i](5));
            quat = w_rot_wodom * quat;
            Eigen::Vector3d trans = w_rot_wodom * odometry_poses[i].block<3, 1>(0, 0) + w_trans_wodom;
            pose.block<3, 1>(0, 0) = trans;
            pose.block<4, 1>(3, 0) = quat.coeffs();
            optimized_poses[odometry_times[i]] = pose;
        }
    }

    void OptimizeRotation() {
        // Warn: The optimized rotations are problematic.
        ceres::Problem rotation_optimizer;
        ceres::Manifold *rotmanifold = new ceres::RotationManifold();
        for (auto &pose : optimized_poses) {
            rotation_optimizer.AddParameterBlock(pose.second.data() + 3, 4, rotmanifold);
        }

        for (size_t i = 0; i < front_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, front_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                std::cerr << "Front pose not found in optimized poses!" << std::endl;
                continue;
            }
            Eigen::Quaterniond quat(front_poses[i](6), front_poses[i](3), front_poses[i](4), front_poses[i](5));      
            SO3Prior* so3prior = new SO3Prior(quat, Eigen::Vector3d(FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma));
            rotation_optimizer.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SO3Prior, 3, 4>(so3prior),
                nullptr, optimized_poses[nearest_time].data() + 3);
        }

        for (size_t i = 0; i < actual_back_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, actual_back_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                continue;
            }
            Eigen::Quaterniond quat_back(back_poses[i](6), back_poses[i](3), back_poses[i](4), back_poses[i](5));
            SO3Prior* so3prior = new SO3Prior(quat_back, Eigen::Vector3d(FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma));
            rotation_optimizer.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SO3Prior, 3, 4>(so3prior),
                nullptr, optimized_poses[nearest_time].data() + 3);
        }

        // add odometry constraints
        for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
            Eigen::Quaterniond quat(odometry_poses[i](6), odometry_poses[i](3), odometry_poses[i](4), odometry_poses[i](5));
            Eigen::Quaterniond quat_next(odometry_poses[i + 1](6), odometry_poses[i + 1](3), odometry_poses[i + 1](4), odometry_poses[i + 1](5));
            Eigen::Quaterniond b1_q_b2 = quat.inverse() * quat_next;

            SO3Edge* so3edge = new SO3Edge(b1_q_b2, Eigen::Vector3d(FLAGS_relative_rot_sigma, FLAGS_relative_rot_sigma, FLAGS_relative_rot_sigma));
            rotation_optimizer.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SO3Edge, 3, 4, 4>(so3edge),
                nullptr, optimized_poses[odometry_times[i]].data() + 3, optimized_poses[odometry_times[i + 1]].data() + 3);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 20;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &rotation_optimizer, &summary);
    }

    void OptimizeTranslation() {
        ceres::Problem translation_optimizer;
        // add prior translations from the front and the back
        for (size_t i = 0; i < front_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, front_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                std::cerr << "Front pose not found in optimized poses!" << std::endl;
                continue;
            }
            PositionPrior* front_prior = new PositionPrior(front_poses[i].block<3, 1>(0, 0),
                    Eigen::Vector3d(FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma));
            translation_optimizer.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PositionPrior, 3, 3>(front_prior), 
                nullptr, optimized_poses[nearest_time].data());
        }
        for (size_t i = 0; i < actual_back_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, actual_back_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                continue;
            }
            PositionPrior* back_prior = new PositionPrior(back_poses[i].block<3, 1>(0, 0), 
                    Eigen::Vector3d(FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma));
            translation_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<PositionPrior, 3, 3>(back_prior), 
                nullptr, optimized_poses[nearest_time].data());
        }
        for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
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

    void OptimizePoseGraph() {
        ceres::Problem pose_graph_optimizer;
        ceres::Manifold *pose_manifold = new ceres::PoseManifoldSimplified();

        for (auto &pose : optimized_poses) {
            pose_graph_optimizer.AddParameterBlock(pose.second.data(), 7, pose_manifold); 
        }

        for (size_t i = 0; i < front_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, front_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                std::cerr << "Front pose not found in optimized poses!" << std::endl;
                continue;
            }
            Eigen::Matrix<double, 6, 1> sigmaPQ;
            sigmaPQ << FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma,
                    FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma;
            EdgeSE3Prior* prior = new EdgeSE3Prior(front_poses[i], sigmaPQ);
            pose_graph_optimizer.AddResidualBlock(prior, nullptr, optimized_poses[nearest_time].data());
        }

        for (size_t i = 0; i < actual_back_times.size(); ++i) {
            auto nearest_time = findNearestTime(optimized_poses, actual_back_times[i], FLAGS_near_time_tol);
            if(optimized_poses.find(nearest_time) == optimized_poses.end()) {
                continue;
            }
            Eigen::Matrix<double, 6, 1> sigmaPQ;
            sigmaPQ << FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma, FLAGS_trans_prior_sigma,
                    FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma, FLAGS_rot_prior_sigma;
            EdgeSE3Prior* prior = new EdgeSE3Prior(back_poses[i], sigmaPQ);
            pose_graph_optimizer.AddResidualBlock(prior, nullptr, optimized_poses[nearest_time].data());
        }
        for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
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
            EdgeSE3* edge = new EdgeSE3(T_ab, sigmaPQ);
            pose_graph_optimizer.AddResidualBlock(edge, nullptr, optimized_poses[odometry_times[i]].data(),
                    optimized_poses[odometry_times[i + 1]].data());
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &pose_graph_optimizer, &summary);
    }

    void saveResults(const std::string &output_file) const {
        outputOptimizedPoses(optimized_poses, output_file); 
    }
}; // class CascadedPgo


int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " odom_file front_loc_file back_loc_file output_path [and other gflags]" << std::endl;
        return 1;
    }
    std::string odometry_file = argv[1];
    std::string front_loc_file = argv[2];
    std::string back_loc_file = argv[3];
    std::string back_loc_folder = back_loc_file.substr(0, back_loc_file.find_last_of("/"));
    std::string back_time_file = back_loc_folder + "/bag_maxtime.txt";
    std::string output_path = argv[4];

    CascadedPgo cpgo(front_loc_file, back_loc_file, odometry_file, back_time_file);
    cpgo.InitializePoses();
    cpgo.saveResults(output_path + "/initial_poses.txt");
    // cpgo.OptimizeRotation();
    // cpgo.saveResults(data_path + "/rotated_poses.txt");
    cpgo.OptimizeTranslation();
    cpgo.saveResults(output_path + "/translated_poses.txt");
    cpgo.OptimizePoseGraph();
    cpgo.saveResults(output_path + "/final_poses.txt");
}
