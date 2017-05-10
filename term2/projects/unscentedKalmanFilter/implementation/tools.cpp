#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::calculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

	VectorXd rmse(4);
	rmse << 0,0,0,0;

	if(estimations.size() != ground_truth.size()
		|| estimations.size() == 0) {
		
		cout << "Invalid estimations or ground truth data" << endl;

		return rmse;
	}

	auto estimations_size = estimations.size();


	for(unsigned int i = 0;  i < estimations_size;  ++i){

		VectorXd residual = estimations[i] - ground_truth[i];

		auto residual_array = residual.array();
		residual = residual_array * residual_array;
		rmse += residual;
	}


	rmse = rmse / estimations_size;
	rmse = rmse.array().sqrt();

	return rmse;

}
