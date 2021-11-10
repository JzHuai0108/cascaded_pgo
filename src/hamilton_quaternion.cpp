#include "hamilton_quaternion.h"

namespace hamilton {
Eigen::Matrix4d quatPlus(const Eigen::Vector4d &q) {
  Eigen::Matrix4d Q;
  Q(0, 0) = q[3];
  Q(0, 1) = -q[2];
  Q(0, 2) = q[1];
  Q(0, 3) = q[0];
  Q(1, 0) = q[2];
  Q(1, 1) = q[3];
  Q(1, 2) = -q[0];
  Q(1, 3) = q[1];
  Q(2, 0) = -q[1];
  Q(2, 1) = q[0];
  Q(2, 2) = q[3];
  Q(2, 3) = q[2];
  Q(3, 0) = -q[0];
  Q(3, 1) = -q[1];
  Q(3, 2) = -q[2];
  Q(3, 3) = q[3];
  return Q;
}

Eigen::Matrix4d quatOPlus(const Eigen::Vector4d &q) {
  Eigen::Matrix4d Q;
  Q(0, 0) = q[3];
  Q(0, 1) = q[2];
  Q(0, 2) = -q[1];
  Q(0, 3) = q[0];
  Q(1, 0) = -q[2];
  Q(1, 1) = q[3];
  Q(1, 2) = q[0];
  Q(1, 3) = q[1];
  Q(2, 0) = q[1];
  Q(2, 1) = -q[0];
  Q(2, 2) = q[3];
  Q(2, 3) = q[2];
  Q(3, 0) = -q[0];
  Q(3, 1) = -q[1];
  Q(3, 2) = -q[2];
  Q(3, 3) = q[3];
  return Q;
}

Eigen::Vector4d qplus(Eigen::Vector4d const &q, Eigen::Vector4d const &p) {
  Eigen::Vector4d qplus_p;
  qplus_p[0] = q[3] * p[0] + q[0] * p[3] + q[1] * p[2] - q[2] * p[1];
  qplus_p[1] = q[3] * p[1] - q[0] * p[2] + q[1] * p[3] + q[2] * p[0];
  qplus_p[2] = q[3] * p[2] + q[0] * p[1] - q[1] * p[0] + q[2] * p[3];
  qplus_p[3] = q[3] * p[3] - q[0] * p[0] - q[1] * p[1] - q[2] * p[2];
  return qplus_p;
}

Eigen::Vector4d qplus2(Eigen::Vector4d const &q, Eigen::Vector4d const &p) {
  Eigen::Map<const Eigen::Quaterniond> sq(q.data());
  Eigen::Map<const Eigen::Quaterniond> sp(p.data());
  Eigen::Vector4d qplus_p;
  Eigen::Map<Eigen::Quaterniond> qp(qplus_p.data());
  qp = sq * sp;
  return qplus_p;
}

Eigen::Matrix3d quat2r(Eigen::Vector4d const & q){
  return Eigen::Map<const Eigen::Quaternion<double>>(q.data()).toRotationMatrix();
}

Eigen::Vector4d quatInv(Eigen::Vector4d const & q)
{
    Eigen::Vector4d qret = q;
    invertQuat(qret);
    return qret;
}

void invertQuat(Eigen::Vector4d & q)
{
    q.head<3>() = -q.head<3>();
}

Eigen::Vector3d qeps(Eigen::Vector4d const & q)
{
    return q.head<3>();
}

Eigen::Vector3f qeps(Eigen::Vector4f const & q)
{
    return q.head<3>();
}

double qeta(Eigen::Vector4d const & q)
{
    return q[3];
}

float qeta(Eigen::Vector4f const & q)
{
    return q[3];
}

Eigen::Vector4d axisAngle2quat(Eigen::Vector3d const & a)
{
    // Method of implementing this function that is accurate to numerical precision from
    // Grassia, F. S. (1998). Practical parameterization of rotations using the exponential map. journal of graphics, gpu, and game tools, 3(3):29–48.

    double theta = a.norm();

    // na is 1/theta sin(theta/2)
    double na;
    if(isLessThenEpsilons4thRoot(theta))
    {
        static const double one_over_48 = 1.0/48.0;
        na = 0.5 + (theta * theta) * one_over_48;
    }
    else
    {
        na = sin(theta*0.5) / theta;
    }
    Eigen::Vector3d axis = a*na;
    double ct = cos(theta*0.5);
    return Eigen::Vector4d(axis[0],axis[1],axis[2],ct);
}

} // namespace hamilton
