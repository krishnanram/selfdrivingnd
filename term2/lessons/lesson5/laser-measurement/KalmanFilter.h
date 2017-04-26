//
// Created by krramasw on 4/25/17.
//

#ifndef LESSON5_KALMANFILTER_H
#define LESSON5_KALMANFILTER_H


#include "../Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class KalmanFilter {
public:

    ///* state vector
    VectorXd x_;

    ///* state covariance matrix
    MatrixXd P_;

    ///* state transistion matrix
    MatrixXd F_;

    ///* process covariance matrix
    MatrixXd Q_;

    ///* measurement matrix
    MatrixXd H_;

    ///* measurement covariance matrix
    MatrixXd R_;

    /**
     * Constructor
     */
    KalmanFilter();

    /**
     * Destructor
     */
    virtual ~KalmanFilter();

    /**
     * Predict Predicts the state and the state covariance
     * using the process model
     */
    void Predict();

    /**
     * Updates the state and
     * @param z The measurement at k+1
     */
    void Update(const VectorXd &z);

};




#endif //LESSON5_KALMANFILTER_H
