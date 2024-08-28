/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Aug 22, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file ImuMeasurements.hpp
 * @brief This file contains the templated measurement structs, structs encapsulating
 *        IMU Sensor data and related typedefs.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_IMU_MEASUREMENTS_HPP_
#define INCLUDE_OKVIS_IMU_MEASUREMENTS_HPP_

#include <deque>
#include <vector>
#include <memory>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#include <Eigen/Dense>
#include <okvis/Time.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/**
 * \brief Generic measurements
 *
 * They always come with a timestamp such that we can perform
 * any kind of asynchronous operation.
 * \tparam MEASUREMENT_T Measurement data type.
 */
template<class MEASUREMENT_T>
struct Measurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::Time timeStamp;      ///< Measurement timestamp
  MEASUREMENT_T measurement;  ///< Actual measurement.
  int sensorId = -1;          ///< Sensor ID. E.g. camera index in a multicamera setup

  /// \brief Default constructor.
  Measurement()
      : timeStamp(0.0) {
  }
  /**
   * @brief Constructor
   * @param timeStamp_ Measurement timestamp.
   * @param measurement_ Actual measurement.
   * @param sensorId Sensor ID (optional).
   */
  Measurement(const okvis::Time& timeStamp_, const MEASUREMENT_T& measurement_,
              int sensorId = -1)
      : timeStamp(timeStamp_),
        measurement(measurement_),
        sensorId(sensorId) {
  }
};

/// \brief IMU measurements. For now assume they are synchronized:
struct ImuSensorReadings {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief Default constructor.
  ImuSensorReadings()
      : gyroscopes(),
        accelerometers() {
  }
  /**
   * @brief Constructor.
   * @param gyroscopes_ Gyroscope measurement.
   * @param accelerometers_ Accelerometer measurement.
   */
  ImuSensorReadings(Eigen::Vector3d gyroscopes_,
                    Eigen::Vector3d accelerometers_)
      : gyroscopes(gyroscopes_),
        accelerometers(accelerometers_) {
  }

  Eigen::Matrix<double, 6, 1> toVector() {
    Eigen::Matrix<double, 6, 1> res;
    res << gyroscopes, accelerometers;
    return res;
  }

  Eigen::Vector3d gyroscopes;     ///< Gyroscope measurement.
  Eigen::Vector3d accelerometers; ///< Accelerometer measurement.
};

// this is how we store raw measurements before more advanced filling into
// data structures happens:
typedef Measurement<ImuSensorReadings> ImuMeasurement;
typedef std::deque<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement> > ImuMeasurementDeque;

}  // namespace okvis

#endif // INCLUDE_OKVIS_IMU_MEASUREMENTS_HPP_
