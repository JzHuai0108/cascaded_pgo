
#ifndef INCLUDE_SWIFTVIO_CERES_IMUERRORWITHGRAVITY_HPP_
#define INCLUDE_SWIFTVIO_CERES_IMUERRORWITHGRAVITY_HPP_

#include <vector>
#include <mutex>
#include <ceres/sized_cost_function.h>
#include <okvis/Time.hpp>
#include <okvis/ImuMeasurements.hpp>
#include <okvis/imu_parameters.hpp>
#include <okvis/Transformation.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

typedef Eigen::Matrix<double, 9, 1> SpeedAndBias;
typedef Eigen::Matrix<double, 9, 1> SpeedAndBiases;

/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Implements a nonlinear IMU factor.
class ImuErrorWithGravity :
    public ::ceres::SizedCostFunction<15 /* number of residuals */,
        7 /* size of first parameter (PoseParameterBlock k) */,
        9 /* size of second parameter (SpeedAndBiasParameterBlock k) */,
        7 /* size of third parameter (PoseParameterBlock k+1) */,
        9 /* size of fourth parameter (SpeedAndBiasParameterBlock k+1) */,
        3 /* gravity direction in the world frame */> {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<15, 7, 9, 7, 9, 3> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 15;

  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 15, 15> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, 15, 15> jacobian_t;

  /// \brief The type of the Jacobian w.r.t. poses --
  /// \warning This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, 15, 7> jacobian0_t;

  /// \brief Default constructor -- assumes information recomputation.
  ImuErrorWithGravity() {
  }

  /// \brief Trivial destructor.
  virtual ~ImuErrorWithGravity() {
  }

  /// \brief Construct with measurements and parameters.
  /// \@param[in] imuMeasurements All the IMU measurements.
  /// \@param[in] imuParameters The parameters to be used.
  /// \@param[in] t_0 Start time.
  /// \@param[in] t_1 End time.
  ImuErrorWithGravity(const okvis::ImuMeasurementDeque & imuMeasurements,
           const okvis::ImuParameters & imuParameters, const okvis::Time& t_0,
           const okvis::Time& t_1);

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @warning This is not actually const, since the re-propagation must somehow be stored...
   * @param[in] T_WS Start pose.
   * @param[in] speedAndBiases Start speed and biases.
   * @return Number of integration steps.
   */
  int redoPreintegration(const okvis::kinematics::Transformation& T_WS,
                         const okvis::SpeedAndBias & speedAndBiases) const;

  // setters

  /// \brief (Re)set the parameters.
  /// \@param[in] imuParameters The parameters to be used.
  void setImuParameters(const okvis::ImuParameters& imuParameters) {
    imuParameters_ = imuParameters;
  }

  /// \brief (Re)set the measurements
  /// \@param[in] imuMeasurements All the IMU measurements.
  void setImuMeasurements(const okvis::ImuMeasurementDeque& imuMeasurements) {
    imuMeasurements_ = imuMeasurements;
  }

  /// \brief (Re)set the start time.
  /// \@param[in] t_0 Start time.
  void setT0(const okvis::Time& t_0) {
    t0_ = t_0;
  }

  /// \brief (Re)set the start time.
  /// \@param[in] t_1 End time.
  void setT1(const okvis::Time& t_1) {
    t1_ = t_1;
  }

  // getters

  /// \brief Get the IMU Parameters.
  /// \return the IMU parameters.
  const okvis::ImuParameters& imuParameters() const {
    return imuParameters_;
  }

  /// \brief Get the IMU measurements.
  const okvis::ImuMeasurementDeque& imuMeasurements() const {
    return imuMeasurements_;
  }

  /// \brief Get the start time.
  okvis::Time t0() const {
    return t0_;
  }

  /// \brief Get the end time.
  okvis::Time t1() const {
    return t1_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  virtual size_t parameterBlocks() const {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const {
    return "ImuErrorWithGravity";
  }

 protected:
  // parameters
  okvis::ImuParameters imuParameters_; ///< The IMU parameters.

  // measurements
  okvis::ImuMeasurementDeque imuMeasurements_; ///< The IMU measurements used. Must be spanning t0_ - t1_.

  // times
  okvis::Time t0_; ///< The start time (i.e. time of the first set of states).
  okvis::Time t1_; ///< The end time (i.e. time of the sedond set of states).

  // preintegration stuff. the mutable is a TERRIBLE HACK, but what can I do.
  // increments (initialise with identity)
  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); ///< Intermediate result
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); ///< Intermediate result
  mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); ///< Intermediate result

  // cross matrix accumulatrion
  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  // sub-Jacobians
  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  /// \brief The Jacobian of the increment (w/o biases).
  mutable Eigen::Matrix<double,15,15> P_delta_ = Eigen::Matrix<double,15,15>::Zero();

  /// \brief Reference biases that are updated when called redoPreintegration.
  mutable SpeedAndBiases speedAndBiases_ref_ = SpeedAndBiases::Zero();

  mutable bool redo_ = true; ///< Keeps track of whether or not this redoPreintegration() needs to be called.
  mutable int redoCounter_ = 0; ///< Counts the number of preintegrations for statistics.

  // information matrix and its square root
  mutable information_t information_; ///< The information matrix for this error term.
  mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.

};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_SWIFTVIO_CERES_IMUERRORWITHGRAVITY_HPP_ */
