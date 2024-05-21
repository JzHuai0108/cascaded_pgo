/**
 * @file pgo.cpp
 * @brief Pose graph optimization
 * Given the absolute pose constraints at the front and the back of a trajectory,
 * and the relative poses between frames from an odometry method,
 * optimize the trajectory to minimize the error in the constraints.
 * This is done in several steps:
 * 1. First only optimize the rotations with a ceres problem;
 * 2. Second only optimize the translations with a ceres problem;
 * 3. Third optimize both rotation and translations with a ceres problem.
 */
#include <pose_factors.h>
#include <ceres/problem.h>

class SO3Prior {
public:
  SO3Prior(const Eigen::Quaterniond &q, const Eigen::Vector3d &sigma_q) : q_(q), sigma_q_(sigma_q) {}

  template <typename T>
  bool operator()(const T *const q, T *residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> q_map(q);
    Eigen::Map<Eigen::Quaternion<T>> residual_map(residual);
    Eigen::AngleAxis<T> aa = Eigen::Quaternion<T>(q_).inverse() * q_map;
    residual_map = aa.angle() * aa.axis();
    residual_map = residual_map / Eigen::Matrix<T, 3, 1>(sigma_q_);
    return true;
  }

private:
    Eigen::Quaterniond q_;
    Eigen::Vector3d sigma_q_;
};

class SO3Edge {
public:
  SO3Edge(const Eigen::Quaterniond &b1_q_b2, const Eigen::Vector3d &sigma_q) : b1_q_b2_(b1_q_b2), sigma_q_(sigma_q) {}

  template <typename T>
  bool operator()(const T *const q1, const T *const q2, T *residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> q1_map(q1);
    Eigen::Map<const Eigen::Quaternion<T>> q2_map(q2);
    Eigen::Map<Eigen::Quaternion<T>> residual_map(residual);
    Eigen::AngleAxis<T> aa = q1_map.inverse() * q2_map * Eigen::Quaternion<T>(b1_q_b2_).inverse();
    residual_map = aa.angle() * aa.axis();
    residual_map = residual_map / Eigen::Matrix<T, 3, 1>(sigma_q_);
    return true;
  }

private:
    Eigen::Quaterniond b1_q_b2_;
    Eigen::Vector3d sigma_q_;
}

class PositionPrior {
public:
    PositionPrior(const Eigen::Vector3d &p, const Eigen::Vector3d &sigma_p) : p_(p), sigma_p_(sigma_p) {}
    
    template <typename T>
    bool operator()(const T *const p, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_map(p);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residual);
        residual_map = p_map - Eigen::Matrix<T, 3, 1>(p_);
        residual_map /= Eigen::Matrix<T, 3, 1>(sigma_p_);
        return true;
    }

private:
    Eigen::Vector3d p_;
    Eigen::Vector3d sigma_p_;
};


class PositionEdge {
public:
  PositionEdge(const Eigen::Vector3d &b1_p_b2, const Eigen::Quaterniond & w_q_b1, const Eigen::Vector3d &sigma_p) : 
    b1_p_b2_(b1_p_b2), w_q_b1_(w_q_b1), sigma_p_(sigma_p) {}

  template <typename T>
  bool operator()(const T *const p, T *residual) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> w_p_b1(p[0]);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> w_p_b2(p[1]);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residual);
    residual_map = w_q_b1_.inverse() * (w_p_b2 - w_p_b1) - Eigen::Matrix<T, 3, 1>(b1_p_b2_);
    residual_map /= Eigen::Matrix<T, 3, 1>(sigma_p_);
    return true;
  }

private:
    Eigen::Vector3d b1_p_b2_;
    Eigen::Quaterniond w_q_b1_;
    Eigen::Vector3d sigma_p_;
};

size_t load_poses(const std::string &posefile,
    std::vector<okvis::Time> &times, std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> &poses, 
    okvis::Duration cull_end_secs = 8) {
    // load poses from file
    std::ofstream stream(posefile);
    if (!stream.is_open()) {
        std::cerr << "Failed to open file " << posefile << std::endl;
        return 1;
    }
    while (!stream.eof()) {
        std::string line;
        std::getline(stream, line);
        std::istringstream iss(line);
        std::string time_str;
        iss >> time_str;
        if (time_str.empty()) {
            break;
        }
        // keep the time precision to nanoseconds
        std::string::size_type pos = time_str.find('.');
        uint32_t sec = std::stoul(time_str.substr(0, pos));
        uint32_t nsec = std::stoul(time_str.substr(pos + 1));
        okvis::Time time(sec, nsec); // TODO: yiwen, copy the okvis time implementation from https://github.com/ethz-asl/okvis/tree/master/okvis_time
        // yiwen: try to keep these okvis time file intact, i.e., minimize the changes to these external files.

        Eigen::Matrix<double, 7, 1> pose;
        for (size_t i = 0; i < 7; ++i) {
            iss >> pose[i];
        }
        times.push_back(time);
        poses.push_back(pose);
    }

    return 0;
}

std::vector<okvis::Time> correct_back_times(std::vector<okvis::Time> &back_times, const okvis::Time &max_bag_time) {
    std::vector<okvis::Time> actual_back_times;
    actual_back_times.reserve(back_times.size());
    for (size_t i = 0; i < back_times.size(); ++i) {
        actual_back_times.push_back(max_bag_time - back_times[i] + max_bag_time);  
    }
}

void load_times(const std::string &timefile, okvis::Time &max_bag_time) {
    // timefile format:
    // first line is max_bag_time,
    // second line is max_bag_time * 2
    std::ifstream stream(timefile);
    if (!stream.is_open()) {
        std::cerr << "Failed to open file " << timefile << std::endl;
        return 1;
    }
    std::string line;
    std::getline(stream, line);
    std::istringstream iss(line);
    std::string time_str;
    iss >> time_str;
    if (time_str.empty()) {
        return 1;
    }
    // keep the time precision to nanoseconds
    std::string::size_type pos = time_str.find('.');
    uint32_t sec = std::stoul(time_str.substr(0, pos));
    uint32_t nsec = std::stoul(time_str.substr(pos + 1));
    max_bag_time = okvis::Time(sec, nsec);
}

int main(int argc, char **argv) {
    std::vector<okvis::Time> front_times;
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> front_poses;

    std::vector<okvis::Time> back_times;
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> back_poses;

    std::vector<okvis::Time> odometry_times;
    std::vector<Eigen::Matrix<double, 7, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>> odometry_poses;

    std::map<okvis::Time, Eigen::Matrix<double, 7, 1>, std::less<okvis::Time>, 
        Eigen::aligned_allocator<std::pair<const okvis::Time, Eigen::Matrix<double, 7, 1>>>> optimized_poses;

    load_poses(front_loc_file, front_times, front_poses);
    load_poses(back_loc_file, back_times, back_poses);
    load_poses(odometry_file, odometry_times, odometry_poses);
    okvis::Time max_bag_time;
    load_times(back_time_file, max_bag_time);

    // put odometry poses into optimized_poses
    for (size_t i = 0; i < odometry_times.size(); ++i) {
        optimized_poses[odometry_times[i]] = odometry_poses[i];
    }

    std::vector<okvis::Time> actual_back_times = correct_back_times(back_times, max_bag_time);

    // rotation optimizer
    ceres::Problem rotation_optimizer;
    // set parameter block parameterization
    ceres::Manifold *rotmanifold = new RotationManifold();
    for (auto &pose : optimized_poses) {
        rotation_optimizer.AddParameterBlock(pose.second.data() + 3, 4, rotmanifold);
    }

    for (size_t i = 0; i < front_times.size(); ++i) {
        SO3Prior prior(front_poses[i].block<4, 1>(3, 0), Eigen::Vector3d(0.1, 0.1, 0.1));
        rotation_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<SO3Prior, 3, 4>(prior), nullptr, optimized_poses[front_times[i]].data() + 3);
    }

    for (size_t i = 0; i < back_times.size(); ++i) {
        SO3Prior prior(back_poses[i].block<4, 1>(3, 0), Eigen::Vector3d(0.1, 0.1, 0.1));
        rotation_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<SO3Prior, 3, 4>(prior), nullptr, optimized_poses[back_times[i]].data() + 3);
    }

    // add odometry constraints
    for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
        SO3Edge edge(odometry_poses[i].block<4, 1>(3, 0).inverse() * odometry_poses[i + 1].block<4, 1>(3, 0), Eigen::Vector3d(0.1, 0.1, 0.1));
        rotation_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<SO3Edge, 3, 4, 4>(edge), nullptr, optimized_poses[odometry_times[i]].data() + 3, optimized_poses[odometry_times[i + 1]].data() + 3);
    }
    rotation_optimizer.optimize();

    // translation optimizer
    ceres::Problem translation_optimizer;

    // add prior translations from the front and the back
    for (size_t i = 0; i < front_times.size(); ++i) {
        PositionPrior prior(front_poses[i].block<3, 1>(0, 0), Eigen::Vector3d(0.1, 0.1, 0.1));
        translation_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<PositionPrior, 3, 3>(prior), nullptr, optimized_poses[front_times[i]].data());
    }
    for (size_t i = 0; i < back_times.size(); ++i) {
        PositionPrior prior(back_poses[i].block<3, 1>(0, 0), Eigen::Vector3d(0.1, 0.1, 0.1));
        translation_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<PositionPrior, 3, 3>(prior), nullptr, optimized_poses[back_times[i]].data());
    }
    for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
        Eigen::Vector3d w_p_b1b2 = odometry_poses[i + 1].block<3, 1>(0, 0) - odometry_poses[i].block<3, 1>(0, 0);
        Eigen::Vector3d b1_p_b2 = odometry_poses[i].block<4, 1>(3, 0).inverse() * w_p_b1b2;
        PositionEdge edge(b1_p_b2, odometry_poses[i].block<4, 1>(3, 0), Eigen::Vector3d(0.1, 0.1, 0.1));
        translation_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<PositionEdge, 3, 3, 3>(edge), nullptr, optimized_poses[odometry_times[i]].data(), optimized_poses[odometry_times[i + 1]].data());
    }
    translation_optimizer.optimize();

    ceres::Problem pose_graph_optimizer;
    // set parameter block parameterization
    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization();
    for (auto &pose : optimized_poses) {
        pose_graph_optimizer.AddParameterBlock(pose.second.data(), 7, quaternion_parameterization);
    }

    for (size_t i = 0; i < front_times.size(); ++i) {
        EdgeSE3Prior prior(front_poses[i]);
        pose_graph_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<EdgeSE3Prior, 6, 7>(prior), nullptr, optimized_poses[front_times[i]].data());
    }
    for (size_t i = 0; i < back_times.size(); ++i) {
        EdgeSE3Prior prior(back_poses[i]);
        pose_graph_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<EdgeSE3Prior, 6, 7>(prior), nullptr, optimized_poses[back_times[i]].data());
    }
    for (size_t i = 0; i < odometry_times.size() - 1; ++i) {
        Eigen::Vector3d w_p_b1b2 = odometry_poses[i + 1].block<3, 1>(0, 0) - odometry_poses[i].block<3, 1>(0, 0);
        Eigen::Vector3d b1_p_b2 = odometry_poses[i].block<4, 1>(3, 0).inverse() * w_p_b1b2;
        EdgeSE3 edge(Eigen::Matrix<double, 7, 1>(b1_p_b2, Eigen::Quaterniond::Identity()), Eigen::Matrix<double, 6, 6>::Identity());
        pose_graph_optimizer.AddResidualBlock(new ceres::AutoDiffCostFunction<EdgeSE3, 6, 7, 7>(edge), nullptr, optimized_poses[odometry_times[i]].data(), optimized_poses[odometry_times[i + 1]].data());
    }

    pose_graph_optimizer.optimize();

    // save the results
    

    return 0;
}