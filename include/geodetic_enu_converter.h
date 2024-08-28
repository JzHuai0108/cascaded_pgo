#pragma once
#include <iostream>
#include <cmath>

#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>

// Function to convert geodetic coordinates to ECEF coordinates
void geodeticToECEF(double phi, double lambda, double h, double &X, double &Y, double &Z);

// Function to convert ECEF to local ENU coordinates
void ecefToENU(double X, double Y, double Z, double X0, double Y0, double Z0, double phi0, double lambda0, double &x, double &y, double &z);

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

/**
 * @brief Convert geodetic coordinates to ENU coordinates
 * @param geodetic Vector of geodetic coordinates (latitude(deg), longitude(deg), altitude(m))
 * @param anchor Anchor point (latitude(deg), longitude(deg), altitude(m))
 * @param[out] enu Vector of ENU coordinates in meters
 */
int geodeticToEnu(const VecVec3d &geodetic,
                  const Eigen::Vector3d &anchor,
                  VecVec3d &enu);
