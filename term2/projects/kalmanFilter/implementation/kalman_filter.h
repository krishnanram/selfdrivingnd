#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "../Eigen/Dense"

class KalmanFilter {
public:

  // state vector
  Eigen::VectorXd stateVector;

  // state covariance matrix
  Eigen::MatrixXd stateCovMatrix;

  // state transistion matrix
  Eigen::MatrixXd stateTranMatrix;

  // process covariance matrix
  Eigen::MatrixXd processCovMatrix;

  // measurement matrix
  Eigen::MatrixXd measurementMatrix;

  // measurement covariance matrix
  Eigen::MatrixXd measurementCovMatrix;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   * @param H_in Measurement matrix
   * @param R_in Measurement covariance matrix
   * @param Q_in Process covariance matrix
   */
  void Init(Eigen::VectorXd &stateVector_in, Eigen::MatrixXd &stateCovMatrix_in_in, Eigen::MatrixXd &stateTranMatrix_in, Eigen::MatrixXd &measurementMatrix_in, Eigen::MatrixXd &measurementCovMatrix_in, Eigen::MatrixXd &processCovMatrix_in);


  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict();

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEKF(const Eigen::VectorXd &z);

};

#endif /* KALMAN_FILTER_H_ */
