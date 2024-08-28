#include "ImuErrorWithGravity.hpp"
#include "pose_manifold.h"
#include "gravity_manifold.h"

#include <thread>

#include <glog/logging.h>

#include <okvis/operators.hpp>
#include <okvis/imu_parameters.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurements and parameters.
ImuErrorWithGravity::ImuErrorWithGravity(const okvis::ImuMeasurementDeque & imuMeasurements,
                   const okvis::ImuParameters & imuParameters,
                   const okvis::Time& t_0, const okvis::Time& t_1) {
  setImuMeasurements(imuMeasurements);
  setImuParameters(imuParameters);
  setT0(t_0);
  setT1(t_1);

  if (t_0 < imuMeasurements.front().timeStamp)
    throw std::runtime_error("First IMU measurement included in ImuErrorWithGravity is not old enough!");
  
  if (t_1 > imuMeasurements.back().timeStamp)
    throw std::runtime_error("Last IMU measurement included in ImuErrorWithGravity is not new enough!");
}

// Propagates pose, speeds and biases with given IMU measurements.
int ImuErrorWithGravity::redoPreintegration(const okvis::kinematics::Transformation& /*T_WS*/,
                                 const okvis::SpeedAndBias & speedAndBiases) const {
  // now the propagation
  okvis::Time time = t0_;
  okvis::Time end = t1_;

  // sanity check:
  assert(imuMeasurements_.front().timeStamp<=time);
  if (!(imuMeasurements_.back().timeStamp >= end))
    return -1;  // nothing to do...

  // increments (initialise with identity)
  Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  C_integral_ = Eigen::Matrix3d::Zero();
  C_doubleintegral_ = Eigen::Matrix3d::Zero();
  acc_integral_ = Eigen::Vector3d::Zero();
  acc_doubleintegral_ = Eigen::Vector3d::Zero();

  // cross matrix accumulatrion
  cross_ = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  dalpha_db_g_ = Eigen::Matrix3d::Zero();
  dv_db_g_ = Eigen::Matrix3d::Zero();
  dp_db_g_ = Eigen::Matrix3d::Zero();

  // the Jacobian of the increment (w/o biases)
  P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

  //Eigen::Matrix<double, 15, 15> F_tot;
  //F_tot.setIdentity();

  double Delta_t = 0;
  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements_.begin();
      it != imuMeasurements_.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = t1_;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t1_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParameters_.sigma_g_c;
    double sigma_a_c = imuParameters_.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParameters_.g_max
        || fabs(omega_S_0[1]) > imuParameters_.g_max
        || fabs(omega_S_0[2]) > imuParameters_.g_max
        || fabs(omega_S_1[0]) > imuParameters_.g_max
        || fabs(omega_S_1[1]) > imuParameters_.g_max
        || fabs(omega_S_1[2]) > imuParameters_.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING)<< "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParameters_.a_max || fabs(acc_S_0[1]) > imuParameters_.a_max
        || fabs(acc_S_0[2]) > imuParameters_.a_max
        || fabs(acc_S_1[0]) > imuParameters_.a_max
        || fabs(acc_S_1[1]) > imuParameters_.a_max
        || fabs(acc_S_1[2]) > imuParameters_.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING)<< "acc saturation";
    }

    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1)
        - speedAndBiases.segment < 3 > (3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = kinematics::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1)
        - speedAndBiases.segment < 3 > (6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_
        + 0.5 * (C + C_1) * acc_S_true * dt;
    // rotation matrix double integral:
    C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
    acc_doubleintegral_ += acc_integral_ * dt
        + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    // Jacobian parts
    dalpha_db_g_ += C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
        + okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_
        + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    dp_db_g_ += dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    // covariance propagation
    Eigen::Matrix<double, 15, 15> F_delta =
        Eigen::Matrix<double, 15, 15>::Identity();
    // transform
#if 0
    F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
        acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
    F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F_delta.block<3, 3>(0, 9) = dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    F_delta.block<3, 3>(0, 12) = -C_integral_ * dt
        + 0.25 * (C + C_1) * dt * dt;
    F_delta.block<3, 3>(3, 9) = -dt * C_1;
    F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
        0.5 * (C + C_1) * acc_S_true * dt);
    F_delta.block<3, 3>(6, 9) = 0.5 * dt
        * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

#else
    Eigen::Matrix3d deltaCross = cross_1- cross_;
    F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
        /*acc_integral_ * dt +*/ 0.25 * (C + C_1) * acc_S_true * dt * dt);
    F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F_delta.block<3, 3>(0, 9) = /*dt * dv_db_g_ + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1)*/
                                0.25*dt*dt*(C_1*acc_S_x*deltaCross);
    F_delta.block<3, 3>(0, 12) = /*-C_integral_ * dt +*/ 0.25 * (C + C_1) * dt * dt;
    F_delta.block<3, 3>(3, 9) = -dt * C_1;
    F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(0.5 * (C + C_1) * acc_S_true * dt);
    F_delta.block<3, 3>(6, 9) = /*0.5 * dt* (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1)*/
                                0.5*dt*(C_1*acc_S_x*deltaCross);
#endif


    F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;
    P_delta_ = F_delta * P_delta_ * F_delta.transpose();
    // add noise. Note that transformations with rotation matrices can be ignored, since the noise is isotropic.
    //F_tot = F_delta*F_tot;
    const double sigma2_dalpha = dt * sigma_g_c
        * sigma_g_c;
    P_delta_(3, 3) += sigma2_dalpha;
    P_delta_(4, 4) += sigma2_dalpha;
    P_delta_(5, 5) += sigma2_dalpha;
    const double sigma2_v = dt * sigma_a_c * sigma_a_c;
    P_delta_(6, 6) += sigma2_v;
    P_delta_(7, 7) += sigma2_v;
    P_delta_(8, 8) += sigma2_v;
    const double sigma2_p = 0.5 * dt * dt * sigma2_v;
    P_delta_(0, 0) += sigma2_p;
    P_delta_(1, 1) += sigma2_p;
    P_delta_(2, 2) += sigma2_p;
    const double sigma2_b_g = dt * imuParameters_.sigma_gw_c * imuParameters_.sigma_gw_c;
    P_delta_(9, 9) += sigma2_b_g;
    P_delta_(10, 10) += sigma2_b_g;
    P_delta_(11, 11) += sigma2_b_g;
    const double sigma2_b_a = dt * imuParameters_.sigma_aw_c * imuParameters_.sigma_aw_c;
    P_delta_(12, 12) += sigma2_b_a;
    P_delta_(13, 13) += sigma2_b_a;
    P_delta_(14, 14) += sigma2_b_a;

    // memory shift
    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == t1_)
      break;

  }

  // store the reference (linearisation) point
  speedAndBiases_ref_ = speedAndBiases;

  // get the weighting:
  // enforce symmetric
  P_delta_ = 0.5 * P_delta_ + 0.5 * P_delta_.transpose().eval();

  // calculate inverse
#if 0
  information_ = P_delta_.inverse();
#else
  information_.setIdentity();
  P_delta_.llt().solveInPlace(information_);
#endif
  information_ = 0.5 * information_ + 0.5 * information_.transpose().eval();

  // square root
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();

  //std::cout << F_tot;

  return i;
}

// This evaluates the error term and additionally computes the Jacobians.
bool ImuErrorWithGravity::Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool ImuErrorWithGravity::EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const {

  // get poses
  const okvis::kinematics::Transformation T_WS_0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));

  const okvis::kinematics::Transformation T_WS_1(
      Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
      Eigen::Quaterniond(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]));

  // get speed and bias
  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (size_t i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  Eigen::Vector3d gravityDirection;
  memcpy(gravityDirection.data(), parameters[4], sizeof(double) * 3);

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  // call the propagation
  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b =
      speedAndBiases_0.tail<6>() - speedAndBiases_ref_.tail<6>();

  redo_ = redo_ || (Delta_b.head<3>().norm() * Delta_t > 0.0001);
  if (redo_) {
    redoPreintegration(T_WS_0, speedAndBiases_0);
    redoCounter_++;
    Delta_b.setZero();
    redo_ = false;
    /*if (redoCounter_ > 1) {
      std::cout << "pre-integration no. " << redoCounter_ << std::endl;
    }*/
  }

  // actual propagation output:
  {
    const Eigen::Vector3d g_W = imuParameters_.g * gravityDirection;

    // assign Jacobian w.r.t. x0
    Eigen::Matrix<double,15,15> F0 =
        Eigen::Matrix<double,15,15>::Identity(); // holds for d/db_g, d/db_a
    const Eigen::Vector3d delta_p_est_W =
        T_WS_0.r() - T_WS_1.r() + speedAndBiases_0.head<3>()*Delta_t + 0.5*g_W*Delta_t*Delta_t;
    const Eigen::Vector3d delta_v_est_W =
        speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>() + g_W*Delta_t;
    const Eigen::Quaterniond Dq = okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b.head<3>())*Delta_q_;
    F0.block<3,3>(0,0) = C_S0_W;
    F0.block<3,3>(0,3) = C_S0_W * okvis::kinematics::crossMx(delta_p_est_W);
    F0.block<3,3>(0,6) = C_S0_W * Eigen::Matrix3d::Identity()*Delta_t;
    F0.block<3,3>(0,9) = dp_db_g_;
    F0.block<3,3>(0,12) = -C_doubleintegral_;
    F0.block<3,3>(3,3) = (okvis::kinematics::plus(Dq*T_WS_1.q().inverse()) *
        okvis::kinematics::oplus(T_WS_0.q())).topLeftCorner<3,3>();
    F0.block<3,3>(3,9) = (okvis::kinematics::oplus(T_WS_1.q().inverse()*T_WS_0.q())*
        okvis::kinematics::oplus(Dq)).topLeftCorner<3,3>()*(-dalpha_db_g_);
    F0.block<3,3>(6,3) = C_S0_W * okvis::kinematics::crossMx(delta_v_est_W);
    F0.block<3,3>(6,6) = C_S0_W;
    F0.block<3,3>(6,9) = dv_db_g_;
    F0.block<3,3>(6,12) = -C_integral_;

    // assign Jacobian w.r.t. x1
    Eigen::Matrix<double,15,15> F1 =
        -Eigen::Matrix<double,15,15>::Identity(); // holds for the biases
    F1.block<3,3>(0,0) = -C_S0_W;
    F1.block<3,3>(3,3) = -(okvis::kinematics::plus(Dq) *
        okvis::kinematics::oplus(T_WS_0.q()) *
        okvis::kinematics::plus(T_WS_1.q().inverse())).topLeftCorner<3,3>();
    F1.block<3,3>(6,6) = -C_S0_W;

    // the overall error vector
    Eigen::Matrix<double, 15, 1> error;
    error.segment<3>(0) =  C_S0_W * delta_p_est_W + acc_doubleintegral_ + F0.block<3,6>(0,9)*Delta_b;
    error.segment<3>(3) = 2*(Dq*(T_WS_1.q().inverse()*T_WS_0.q())).vec(); //2*T_WS_0.q()*Dq*T_WS_1.q().inverse();//
    error.segment<3>(6) = C_S0_W * delta_v_est_W + acc_integral_ + F0.block<3,6>(6,9)*Delta_b;
    error.tail<6>() = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();

    // error weighting
    Eigen::Map<Eigen::Matrix<double, 15, 1> > weighted_error(residuals);
    weighted_error = squareRootInformation_ * error;

    // get the Jacobians
    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, 15, 6> J0_minimal = squareRootInformation_
            * F0.block<15, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        ::ceres::PoseManifold::liftJacobian(parameters[0], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J0(
            jacobians[0]);
        J0 = J0_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J0_minimal_mapped(
                jacobiansMinimal[0]);
            J0_minimal_mapped = J0_minimal;
          }
        }
      }
      if (jacobians[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J1(
            jacobians[1]);
        J1 = squareRootInformation_ * F0.block<15, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J1_minimal_mapped(
                jacobiansMinimal[1]);
            J1_minimal_mapped = J1;
          }
        }
      }
      if (jacobians[2] != NULL) {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, 15, 6> J2_minimal = squareRootInformation_
                    * F1.block<15, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        ::ceres::PoseManifold::liftJacobian(parameters[2], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J2(
            jacobians[2]);
        J2 = J2_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[2] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J2_minimal_mapped(
                jacobiansMinimal[2]);
            J2_minimal_mapped = J2_minimal;
          }
        }
      }
      if (jacobians[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J3(jacobians[3]);
        J3 = squareRootInformation_ * F1.block<15, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[3] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J3_minimal_mapped(
                jacobiansMinimal[3]);
            J3_minimal_mapped = J3;
          }
        }
      }
      if (jacobians[4] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J4(
            jacobians[4]);
        Eigen::Matrix<double, 15, 3> de_dunitgW =
            Eigen::Matrix<double, 15, 3>::Zero();
        de_dunitgW.topLeftCorner<3, 3>() = 0.5 * Delta_t * Delta_t *
                                           imuParameters_.g *
                                           C_S0_W;
        de_dunitgW.block<3, 3>(6, 0) = Delta_t * imuParameters_.g * C_S0_W;
        J4 = squareRootInformation_ * de_dunitgW;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[4] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 2, Eigen::RowMajor>>
                J4_minimal_mapped(jacobiansMinimal[4]);
            Eigen::Matrix<double, 3, 2, Eigen::RowMajor> dunitgW_du;
            ::ceres::UnitVec3Manifold::plusJacobian(
                gravityDirection.data(), dunitgW_du.data());
            J4_minimal_mapped = J4 * dunitgW_du;
          }
        }
      }
    }
  }
  return true;
}

}  // namespace ceres
}  // namespace okvis
