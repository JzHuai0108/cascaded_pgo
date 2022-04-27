// Given consecutive poses obtained by a drifting odometer,
// and sparse key poses obtained by an optimization backend,
// densify the sparse poses with the dense consecutive odometry poses,
// by using a pose graph optimization on SE3.

#include <cstdlib>
#include <fstream>
#include <iomanip> // std::setprecision
#include <iostream>
#include <string>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "pose_factors.h"
#include "pose_local_parameterization.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

/**
 * @brief loadcsv
 * @param datafile
 * @param res [time(sec) tx ty tz qx qy qz qw] list
 */
void loadcsv(const std::string &datafile,
             std::vector<std::vector<double>> &res, char delim=' ') {
  std::ifstream infile(datafile);
  std::string line;
  std::getline(infile, line);
  while (std::getline(infile, line)) {
    std::istringstream sin(line);
    double d;
    std::vector<double> temp;
    char tempdelim;
    while (sin >> d) {
      temp.push_back(d);
      if (delim != ' ') {
        sin >> tempdelim;
      }
    }
    res.push_back(temp);
  }
  infile.close();
}

/**
 * @brief savecsv
 * @param outfile
 * @param data [time(sec) tx ty tz qx qy qz qw] list
 * @param fileHead
 */
void savecsv(std::string outfile, const std::vector<std::vector<double>> &data,
             std::string fileHead) {
  std::ofstream ofs(outfile, std::ofstream::out);
  ofs << fileHead << std::endl;
  for (size_t i = 0; i < data.size(); i++) {
    ofs << std::setprecision(18) << std::fixed << data[i][0] << " ";
    for (size_t j = 1; j < data[0].size(); j++) {
      ofs << std::setprecision(12) << std::fixed << data[i][j] << " ";
    }
    ofs << std::endl;
  }
  ofs.close();
}

struct AssociatedPose {
  size_t kfId;
  size_t fId;
  double timeDiff;

  AssociatedPose(size_t _kfId, size_t _fId, double _timeDiff) : kfId(_kfId), fId(_fId), timeDiff(_timeDiff) {}

};

/**
 * @brief associatePoses
 * @param f_poses
 * @param kf_poses
 * @param kf_indices the index of the frame pose closest in time to the keyframe pose.
 */
void associatePoses(const std::vector<std::vector<double>> &f_poses,
                    const std::vector<std::vector<double>> &kf_poses,
                    double timetolerance,
                    std::vector<AssociatedPose> *kf_indices) {
  kf_indices->clear();
  size_t lastUsedFrameId = 0;
  kf_indices->reserve(kf_poses.size());
  for (size_t i = 0; i < kf_poses.size(); ++i) {
     double kf_time = kf_poses[i][0];
     double minTimeDiff = 1000;
     size_t minId = 0;
     size_t j = lastUsedFrameId;

     for (; j < f_poses.size(); ++j) {
        double diff = std::fabs(f_poses[j][0] - kf_time);
        if (diff < minTimeDiff) {
          minTimeDiff = diff;
          minId = j;
        }
     }
     if (minTimeDiff < timetolerance) {
       kf_indices->emplace_back(i, minId, minTimeDiff);
       lastUsedFrameId = minId;
     }
  }

  // check
  for (const auto &ap : *kf_indices) {
    CHECK_LT(ap.timeDiff, timetolerance);
    CHECK_LT(ap.fId, f_poses.size());
  }

  LOG(INFO) << "Associated #keyframe poses " << kf_indices->size()
            << " to frames out of total #keyframe poses " << kf_poses.size()
            << ". If the number of associations is too few, try to increase "
               "time diff "
               "toleranace (current value "
            << timetolerance << " sec).";
}

void densifyPoses(const std::string &framePoseTxt,
                  const std::string &keyframePoseTxt,
                  const std::string &outfile, double timetolerance) {
  std::cout << "Frame pose file " << framePoseTxt << "\nKeyframe pose file "
            << keyframePoseTxt << "\n";
  std::vector<std::vector<double>> kf_poses, f_poses;
  // Note that frames includes all keyframes.
//  loadcsv(framePoseTxt, f_poses, ',');
  loadcsv(framePoseTxt, f_poses, ' ');
  CHECK_GT(f_poses.size(), 0) << "No. pose loaded from " << framePoseTxt << ".";
  CHECK_GE(f_poses.front().size(), 8)
      << "Too few columns (" << f_poses.front().size() << ") loaded from "
      << framePoseTxt
      << ". Ensure that the lines of the file agrees with TUM format.";
  loadcsv(keyframePoseTxt, kf_poses);

  const int SIZE_POSE = 7;
  std::vector<Eigen::Matrix<double, 7, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
      estimated_poses;
  estimated_poses.reserve(f_poses.size());

  // 2.1 initialization of SE3 poses: for keyframes, use the values loaded
  // from the file, for frames, initialize their poses relative to the nearest
  // keyframe using relative motion.
  for (size_t i = 0u; i < f_poses.size(); i++) {
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> pose(&f_poses[i][1]);
    estimated_poses.push_back(pose);
  }

  std::vector<AssociatedPose> kf_indices;
  associatePoses(f_poses, kf_poses, timetolerance, &kf_indices);

  Problem problem;
  ceres::LocalParameterization *local_parameterization =
      new PoseLocalParameterization();
  for (size_t i = 0; i < f_poses.size(); i++) {
    problem.AddParameterBlock(estimated_poses[i].data(), SIZE_POSE,
                              local_parameterization);
  }

  // 2.2 observations:
  // SE3 pose priors for keyframes
  for (size_t i = 0u; i < kf_indices.size(); i++) {
    Eigen::Map<Eigen::Matrix<double, 7, 1>> pose(&kf_poses[kf_indices[i].kfId][1]);

    CostFunction *cost_function = new EdgeSE3Prior(pose);
    problem.AddResidualBlock(cost_function, nullptr,
                             estimated_poses[kf_indices[i].fId].data());
  }

  // SE3 relative motion constraints for consecutive frames
  for (size_t i = 0; i < f_poses.size() - 1; i++) {
    Eigen::Quaterniond q1(f_poses[i][7], f_poses[i][4], f_poses[i][5],
                          f_poses[i][6]);
    Eigen::Vector3d p1(f_poses[i][1], f_poses[i][2], f_poses[i][3]);

    Eigen::Quaterniond q2(f_poses[i + 1][7], f_poses[i + 1][4],
                          f_poses[i + 1][5], f_poses[i + 1][6]);
    Eigen::Vector3d p2(f_poses[i + 1][1], f_poses[i + 1][2], f_poses[i + 1][3]);

    Eigen::Matrix<double, 7, 1> T_12;
    Eigen::Map<Eigen::Quaterniond> q_12(T_12.data() + 3);
    q_12 = q1.inverse() * q2;
    Eigen::Map<Eigen::Vector3d> p_12(T_12.data());
    p_12 = q1.inverse() * (p2 - p1);

    CostFunction *cost_function = new EdgeSE3(T_12);
    problem.AddResidualBlock(cost_function, nullptr, estimated_poses[i].data(),
                             estimated_poses[i + 1].data());
  }

  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  // 3. save the refined frame poses to a csv file.
  for (size_t i = 0u; i < f_poses.size(); i++) {
    Eigen::Map<Eigen::Quaterniond> q(estimated_poses[i].data() + 3);
    Eigen::Map<Eigen::Vector3d> t(estimated_poses[i].data());
    f_poses[i][1] = t(0);
    f_poses[i][2] = t(1);
    f_poses[i][3] = t(2);
    f_poses[i][4] = q.x();
    f_poses[i][5] = q.y();
    f_poses[i][6] = q.z();
    f_poses[i][7] = q.w();
  }
  std::string fileHead = "# timestamp tx ty tz qx qy qz qw";
  savecsv(outfile, f_poses, fileHead);
  std::cout << "Saved densified poses to " << outfile << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " frame_poses.txt keyframe_poses.txt\n"
              << "Both files should have lines in TUM RGBD format, i.e., "
                 "time[sec] tx ty tz qx qy qz qw\n";
    return -1;
  }
  std::string framePoseTxt = argv[1];
  std::string keyframePoseTxt = argv[2];
  size_t pos = framePoseTxt.find_last_of("/\\");
  std::string outfile = framePoseTxt.substr(0, pos) + "/dense_poses.txt";

  if (argc > 3) {
    outfile = argv[3];
  }
  double timetolerance = 2e-3;
  if (argc > 4) {
    timetolerance = std::atof(argv[4]);
  }
  densifyPoses(framePoseTxt, keyframePoseTxt, outfile, timetolerance);
  return 0;
}
