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
#include <glog/logging.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/types/slam3d_addons/types_slam3d_addons.h"

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

inline Eigen::Isometry3d toIsometry(const Eigen::Quaterniond &q, const Eigen::Vector3d &t) {
  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
  iso.linear() = q.toRotationMatrix();
  iso.translation() = t;
  return iso;
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

  // 2. build the g2o optimization problem, refer to optimizeScaleTrans in
  // scale_solver.cpp of this package, OR
  // https://github.com/JzHuai0108/sim3opt/blob/master/kitti_surf.cpp#L542
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(true);
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver =
      g2o::make_unique<
          g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver =
      new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
  optimizer.setAlgorithm(solver);

  // 2.1 states: SE3 poses for all frames. Frames includes all keyframes.
  // These states are represented by G2oVertexSE3
  // SET  VERTICES
  size_t count = 0;
  std::vector<int> kf_indices;

  for (size_t i = 0u; i < f_poses.size(); i++) {
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(i);

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
      v->setEstimate(toIsometry(refKeyframeQuat, refKeyframeTrans));
      count++;
    } else {
      Eigen::Quaterniond q(f_poses[count][7], f_poses[count][4],
                           f_poses[count][5], f_poses[count][6]);
      Eigen::Vector3d t(f_poses[count][1], f_poses[count][2],
                        f_poses[count][3]);
      v->setEstimate(toIsometry(q, t));
    }

    optimizer.addVertex(v);
  }
  CHECK_EQ(kf_indices.size(), kf_poses.size())
      << "Found keyframes in frames " << count << " and keyframe poses "
      << kf_poses.size() << ".";

  // 2.2 observations: // SET EDGES
  // SE3 pose priors for keyframes, represented by G2oSE3Observation6DEdge
  for (size_t i = 0u; i < kf_poses.size(); i++) {
    g2o::EdgeSE3Prior *e = new g2o::EdgeSE3Prior();
    e->setVertex(0, optimizer.vertex(kf_indices[i]));

    Eigen::Quaterniond q(kf_poses[i][7], kf_poses[i][4], kf_poses[i][5],
                         kf_poses[i][6]);
    Eigen::Vector3d t(kf_poses[i][1], kf_poses[i][2], kf_poses[i][3]);
    e->setMeasurement(toIsometry(q, t));
    LOG(INFO) << "key " << i << " " << q.coeffs().transpose() << " t " << t.transpose();
    Eigen::Matrix<double, 6, 6> infoMat;
    infoMat.setIdentity();
    e->setInformation(infoMat);
    e->setParameterId(0, 0);
    optimizer.addEdge(e);
  }
  // SE3 relative motion constraints for consecutive frames, represented by
  // G2oEdgeSE3
  for (size_t i = 0; i < f_poses.size() - 1; i++) {
    g2o::EdgeSE3 *e = new g2o::EdgeSE3();
    e->setVertex(0, optimizer.vertex(i));
    e->setVertex(1, optimizer.vertex(i + 1));

    Eigen::Quaterniond q1(f_poses[i][7], f_poses[i][4], f_poses[i][5],
                          f_poses[i][6]);
    Eigen::Vector3d t1(f_poses[i][1], f_poses[i][2], f_poses[i][3]);
    Eigen::Isometry3d Ta = toIsometry(q1, t1);
    Eigen::Quaterniond q2(f_poses[i + 1][7], f_poses[i + 1][4],
                          f_poses[i + 1][5], f_poses[i + 1][6]);
    Eigen::Vector3d t2(f_poses[i + 1][1], f_poses[i + 1][2], f_poses[i + 1][3]);
    Eigen::Isometry3d Tb = toIsometry(q2, t2);
    e->setMeasurement(Ta.inverse() * Tb);

    Eigen::Matrix<double, 6, 6> infoMat;
    infoMat.setIdentity();
    e->setInformation(infoMat);

    optimizer.addEdge(e);
  }
  LOG(INFO) << "Initialize optimization!";
  optimizer.initializeOptimization();
  // 2.4 optimize the poses
  LOG(INFO) << "Optimize.";
  optimizer.optimize(3);
  LOG(INFO) << "Save!";
  // 3. save the refined frame poses to a csv file with the same format as the
  // poses.txt.
  for (size_t i = 0u; i < f_poses.size(); i++) {
    g2o::VertexSE3 *v =
        static_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
    Eigen::Matrix3d R = v->estimate().linear();
    Eigen::Quaterniond q(R);
    Eigen::Vector3d t = v->estimate().translation();
    f_poses[i][1] = t(0);
    f_poses[i][2] = t(1);
    f_poses[i][3] = t(2);
    f_poses[i][4] = q.x();
    f_poses[i][5] = q.y();
    f_poses[i][6] = q.z();
    f_poses[i][7] = q.w();
  }
  std::string outfile = "./densify_res.txt";
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
