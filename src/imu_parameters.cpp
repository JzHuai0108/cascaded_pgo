#include <okvis/imu_parameters.hpp>

namespace okvis {

ImuParameters createX36DImuParameters() {
  ImuParameters params = ImuParameters();
  params.g = 9.79354; // wuhan earth surface local gravity
  params.sigma_g_c = 1.2e-3;
  params.sigma_a_c = 8e-3;
  params.sigma_bg = 0.03;
  params.sigma_ba = 0.1;
  params.sigma_gw_c = 4e-6;
  params.sigma_aw_c = 4e-5;
  params.rate = 100;
  params.model_name = "BG_BA";
  return params;
}

ImuParameters::ImuParameters()
    : a_max(200.0),
      g_max(10),
      sigma_g_c(1.2e-3),
      sigma_a_c(8e-3),
      sigma_bg(0.03),
      sigma_ba(0.1),
      sigma_gw_c(4e-6),
      sigma_aw_c(4e-5),
      tau(3600.0),
      g(9.80665),
      rate(100),
      sigma_Mg_element(0.0),
      sigma_Ts_element(0.0),
      sigma_Ma_element(0.0),
      imuIdx(0),
      model_name("BG_BA_MG_TS_MA"),
      sigma_gravity_direction(0.0),
      g0(0, 0, 0),
      a0(0, 0, 0),
      normalGravity(0, 0, -1) {
  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Mg0 = eye;
  Ts0.setZero();
  Ma0 << 1, 0, 1, 0, 0, 1;
}

const Eigen::Vector3d &ImuParameters::gravityDirection() const {
  return normalGravity;
}

Eigen::Vector3d ImuParameters::gravity() const {
  return g * normalGravity;
}

void ImuParameters::setGravityDirection(
    const Eigen::Vector3d &gravityDirection) {
  normalGravity = gravityDirection;
}

std::string ImuParameters::toString(const std::string &hint) const {
  std::stringstream ss;
  ss << hint << "a max " << a_max << ", g max " << g_max << ", sigma_g_c " << sigma_g_c
            << ", sigma_a_c " << sigma_a_c << ", sigma_gw_c " << sigma_gw_c << ", sigma_aw_c "
            << sigma_aw_c << ".\n";
  ss << "sigma_bg " << sigma_bg << ", sigma ba " << sigma_ba << ", g " << g << " unit gravity "
     << normalGravity.transpose() << ",\nsigma gravity direction " << sigma_gravity_direction
     << ".\n";

  ss << "rate " << rate << ", imu idx " << imuIdx << ", imu model " << model_name << ".\n";
  ss << "sigma_Mg_element " << sigma_Mg_element << ", sigma_Ts_element " << sigma_Ts_element
     << ", sigma_Ma_element " << sigma_Ma_element << ".\n";
  ss << "g0 " << g0.transpose() << ", a0 " << a0.transpose() << ".\nMg0 " << Mg0.transpose()
     << ".\nTs0 " << Ts0.transpose() << ".\nMa0 " << Ma0.transpose() << ".\n";
  return ss.str();
}
} // namespace okvis