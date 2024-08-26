#include "geodetic_enu_converter.h"

#include <fstream>
#include <iomanip>

// Define Earth's constants
const double a = 6378137.0;           // WGS-84 Earth semimajor axis (meters)
const double f = 1.0 / 298.257223563; // WGS-84 flattening factor
const double b = a * (1 - f);         // Semi-minor axis
const double e2 = 1 - (b * b) / (a * a); // Square of eccentricity

// Function to convert geodetic coordinates to ECEF coordinates
void geodeticToECEF(double phi, double lambda, double h, double &X, double &Y, double &Z) {
    double sin_phi = sin(phi);
    double cos_phi = cos(phi);
    double sin_lambda = sin(lambda);
    double cos_lambda = cos(lambda);

    // Radius of curvature in the prime vertical
    double N = a / sqrt(1 - e2 * sin_phi * sin_phi);

    // Calculate ECEF coordinates
    X = (N + h) * cos_phi * cos_lambda;
    Y = (N + h) * cos_phi * sin_lambda;
    Z = (N * (1 - e2) + h) * sin_phi;
}

// Function to convert ECEF to local ENU coordinates
void ecefToENU(double X, double Y, double Z, double X0, double Y0, double Z0, double phi0, double lambda0, double &x, double &y, double &z) {
    // Differences in ECEF coordinates
    double dX = X - X0;
    double dY = Y - Y0;
    double dZ = Z - Z0;

    // Calculate ENU coordinates using the rotation matrix
    x = -sin(lambda0) * dX + cos(lambda0) * dY;
    y = -sin(phi0) * cos(lambda0) * dX - sin(phi0) * sin(lambda0) * dY + cos(phi0) * dZ;
    z = cos(phi0) * cos(lambda0) * dX + cos(phi0) * sin(lambda0) * dY + sin(phi0) * dZ;
}

int geodeticToEnu(const VecVec3d &geodetic,
                  const Eigen::Vector3d &anchor,
                  VecVec3d &enu) {
    // Define the reference point (anchor) in degrees and meters
    double phi0 = anchor[0] * M_PI / 180.0; // Reference latitude in radians
    double lambda0 = anchor[1] * M_PI / 180.0; // Reference longitude in radians
    double h0 = anchor[2]; // Reference altitude in meters

    enu.resize(geodetic.size());
    // Convert reference and target points to ECEF coordinates
    double X0, Y0, Z0;
    geodeticToECEF(phi0, lambda0, h0, X0, Y0, Z0);
    for (size_t i = 0; i < geodetic.size(); ++i) {
        double X, Y, Z;
        geodeticToECEF(geodetic[i][0] * M_PI / 180.0, geodetic[i][1] * M_PI / 180.0, geodetic[i][2], X, Y, Z);

        // Convert ECEF to ENU coordinates
        double x, y, z;
        ecefToENU(X, Y, Z, X0, Y0, Z0, phi0, lambda0, x, y, z);

        enu[i] = Eigen::Vector3d(x, y, z);
    }
    return 0;
}
