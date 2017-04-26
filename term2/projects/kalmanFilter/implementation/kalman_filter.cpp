#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &stateVector_in, MatrixXd &stateCovMatrix_in, MatrixXd &stateTranMatrix_in,
                        MatrixXd &measurementMatrix_in, MatrixXd &measurementCovMatrix_in, MatrixXd &processCovMatrix_in) {

// VectorXd &x_in,    stateVector_in
// MatrixXd &P_in,    stateCovMatrix_in
// MatrixXd &F_in,    stateTranMatrix_in
// MatrixXd &H_in,    measurementMatrix_in
// MatrixXd &R_in,    measurementCovMatrix_in
// MatrixXd &Q_in     processCovMatrix_in

  stateVector = stateVector_in;
  stateCovMatrix = stateCovMatrix_in;

  measurementMatrix = measurementMatrix_in;
  measurementCovMatrix = measurementCovMatrix_in;
  processCovMatrix = processCovMatrix_in;

  stateTranMatrix = stateTranMatrix_in;
}

void KalmanFilter::Predict() {
  /*
   * Predict the state
   */

  stateVector = (stateTranMatrix * stateVector);

  MatrixXd F_transpose = stateTranMatrix.transpose();
  stateCovMatrix = stateTranMatrix * stateCovMatrix * F_transpose + processCovMatrix;

}

void KalmanFilter::Update(const VectorXd &z) {
  /*
   * Update the state
   */

  VectorXd z_pred = measurementMatrix * stateVector;
  VectorXd y_ = z - z_pred;  // new filter for error calculation

  MatrixXd H_transpose = measurementMatrix.transpose();
  MatrixXd P_h_transpose = stateCovMatrix * H_transpose;

  MatrixXd S_ = measurementMatrix * P_h_transpose + measurementCovMatrix;
  MatrixXd S_inverse = S_.inverse();

  MatrixXd K_ = P_h_transpose * S_inverse;

  // new estimate
  stateVector = stateVector + (K_ * y_);
  long x_size = stateVector.size();
  MatrixXd I_ = MatrixXd::Identity(x_size, x_size);
  stateCovMatrix = (I_ - K_ * measurementMatrix) * stateCovMatrix;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /*
   * update the state by using Extended Kalman Filter equations
   */


  VectorXd new_polar(3);
  new_polar[0] = sqrt(stateVector[0] * stateVector[0] + stateVector[1] * stateVector[1]);
  new_polar[1] = atan2(stateVector[1], stateVector[0]);

  double d = new_polar[0];
  if (d < 1e-6) d = 1e-6;

  new_polar[2] = (stateVector[0] * stateVector[2] + stateVector[1] * stateVector[3]) / d;

  VectorXd z_pred = new_polar;
  VectorXd y_ = z - z_pred;

  // same as normal update
  MatrixXd H_transpose = measurementMatrix.transpose();
  MatrixXd S_ = measurementMatrix * stateCovMatrix * H_transpose + measurementCovMatrix;

  MatrixXd S_inverse = S_.inverse();

  MatrixXd P_h_transpose = stateCovMatrix * H_transpose;
  MatrixXd K_ = P_h_transpose * S_inverse;

  // new estimate
  stateVector = stateVector + (K_ * y_);
  long x_size = stateVector.size();
  MatrixXd I_ = MatrixXd::Identity(x_size, x_size);
  stateCovMatrix = (I_ - K_ * measurementMatrix) * stateCovMatrix;


}