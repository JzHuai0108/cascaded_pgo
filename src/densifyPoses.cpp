// Given consecutive poses obtained by a drifting odometer,
// and sparse key poses obtained by an optimization backend,
// densify the sparse poses with the dense consecutive odometry poses,
// by using a pose graph optimization on SE3.

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

void loadcsv(const std::string &datafile,
             std::vector<std::vector<double>> &res) {
  std::ifstream infile(datafile);
  std::string line;
  std::getline(infile, line);
  while (std::getline(infile, line)) {
    std::istringstream sin(line);
    double d;
    std::vector<double> temp;
    while (sin >> d) {
      temp.push_back(d);
    }
    res.push_back(temp);
  }
  infile.close();
}

void savecsv(std::string outfile, std::vector<std::vector<double>> &data,
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

void densifyPoses(const std::string &framePoseTxt,
                  const std::string &keyframePoseTxt) {
  std::cout << "Frame pose file " << framePoseTxt << "\nKeyframe pose file "
            << keyframePoseTxt << "\n";
  // 1.1 load keyframe poses from test/data/dataset-corridor1_512_16_poses.txt,
  // refer to loadCsvData in SimDataInterface.cpp of repo swift_vio.
  std::vector<std::vector<double>> kf_poses, f_poses;
  loadcsv(keyframePoseTxt, kf_poses);
  // 1.2 load frame poses from test/data/dataset-corridor1_512_16_kfposes.txt,
  // refer to loadCsvData in SimDataInterface.cpp of repo swift_vio.
  loadcsv(framePoseTxt, f_poses);

  // 2.1 states: SE3 poses for all frames. Frames includes all keyframes.
  // These states are represented by G2oVertexSE3
  // SET  VERTICES
  size_t count = 0;
  std::vector<int> kf_indices;
  const int SIZE_POSE = 7;

  std::vector<Eigen::Matrix<double, 7, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
      estimated_poses;

  for (size_t i = 0u; i < f_poses.size(); i++) {
    // 2.1.1 initialization of SE3 poses: for keyframes, use the values loaded
    // from the file, for frames, initialize their poses relative to the nearest
    // keyframe using relative motion.
    if (count < kf_poses.size() && f_poses[i][0] == kf_poses[count][0]) {
      // v->setFixed(true);
      kf_indices.push_back(i);
      Eigen::Quaterniond refKeyframeQuat(kf_poses[count][7], kf_poses[count][4],
                                         kf_poses[count][5],
                                         kf_poses[count][6]);
      Eigen::Vector3d refKeyframeTrans(kf_poses[count][1], kf_poses[count][2],
                                       kf_poses[count][3]);

      Eigen::Map<Eigen::Matrix<double, 7, 1>> pose(&kf_poses[count][1]);
      estimated_poses.push_back(pose);
      count++;
    } else {
      Eigen::Map<Eigen::Matrix<double, 7, 1>> pose(&f_poses[i][1]);
      estimated_poses.push_back(pose);
    }
  }
  CHECK_EQ(kf_indices.size(), kf_poses.size())
      << "Found keyframes in frames " << count << " and keyframe poses "
      << kf_poses.size() << ".";
  CHECK_EQ(f_poses.size(), estimated_poses.size())
      << "Frame poses " << f_poses.size() << " and initial poses "
      << estimated_poses.size() << ".";

  Problem problem;
  ceres::LocalParameterization *local_parameterization =
      new PoseLocalParameterization();
  for (size_t i = 0; i < f_poses.size(); i++) {
    problem.AddParameterBlock(estimated_poses[i].data(), SIZE_POSE,
                              local_parameterization);
  }

  // 2.2 observations: // SET EDGES
  // SE3 pose priors for keyframes
  for (size_t i = 0u; i < kf_poses.size(); i++) {
    Eigen::Map<Eigen::Matrix<double, 7, 1>> pose(&kf_poses[i][1]);

    CostFunction *cost_function = new EdgeSE3Prior(pose);
    problem.AddResidualBlock(cost_function, nullptr,
                             estimated_poses[kf_indices[i]].data());
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

  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  // 3. save the refined frame poses to a csv file with the same format as the
  // poses.txt.
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
  size_t pos = framePoseTxt.find_last_of("/\\");

  std::string outfile = framePoseTxt.substr(0, pos) + "/dense_poses.txt";

  LOG(INFO) << "Saving densified poses to " << outfile;
  std::string fileHead = "# timestamp tx ty tz qx qy qz qw";
  savecsv(outfile, f_poses, fileHead);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " frame_poses.txt keyframe_poses.txt\n"
              << "Both files should have lines in TUM RGBD format, i.e., "
                 "time[sec] tx ty tz qx qy qz qw\n";
    return -1;
  }
  densifyPoses(argv[1], argv[2]);
  return 0;
}
