/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "ParticleFilter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 300;
	particles.resize(num_particles);

	//  initialize with normal (Gaussian) distribution for x, y and theta.

	std::default_random_engine gen;

	std::normal_distribution<double> xDist(0, std[0]);
	std::normal_distribution<double> yDist(0, std[1]);
	std::normal_distribution<double> thetaDist(0, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		particles[i].id = i;
		particles[i].x = x + xDist(gen);
		particles[i].y = y + yDist(gen);
		particles[i].theta = theta + thetaDist(gen);

		particles[i].weight = 1.0;  		// set weights to 1
	}
	is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yawRate) {


    /*
     *  Step 1:     create Gaussian noise distribution for contol data
     *  Step 2: 	for each particle, estimate the next state using velocity and yawRate
     *
     */

	std::default_random_engine gen;

	// create noise distribution
	std::normal_distribution<double> xDist(0, std_pos[0]);
	std::normal_distribution<double> yDist(0, std_pos[1]);
	std::normal_distribution<double> thetaDist(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++){

		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		double vyawRate = velocity / yawRate;
		double yaw_dt 	= yawRate * delta_t;
		double v_dt 	= velocity * delta_t;

		// prediction
		if (fabs(yawRate) > 1e-5){
			x += vyawRate*(sin(theta + yaw_dt) - sin(theta));
			y += vyawRate*(cos(theta) - cos(theta + yaw_dt));
			theta += yaw_dt;
		}
		else{

			// car is moving straight ...
			x += v_dt*sin(theta);
			y += v_dt*cos(theta);

		}

		particles[i].id = i;
		particles[i].x = x + xDist(gen);
		particles[i].y = y + yDist(gen);
		particles[i].theta = theta + thetaDist(gen);
	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {


	/*
	 * 	Goal : Find the predicted measurement that is closest to each observed measurement and assign the
	 * 		observed measurement to this particular landmark.
	 *
     *  for each observation in observations
     * 		for each landmark
     *  		measure the distance; check for minimum distance
     *
    */


	for (int obsIndex = 0; obsIndex < observations.size(); obsIndex++) {

		int corresponding_id = -1;
		double min_distance = std::numeric_limits<double>::max();

		for (unsigned int predIndex = 0; predIndex < predicted.size(); predIndex++) {
			double distance = dist(observations[obsIndex].x, observations[obsIndex].y,
								   predicted[predIndex].x, predicted[predIndex].y);

			if (distance < min_distance){
				min_distance = distance;
				corresponding_id = predIndex;
			}
		}
		observations[obsIndex].id = corresponding_id;
	}


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	/* Logic
	 *
	 * Update the weights of each particle using a mult-variate Gaussian distribution.
	 *
	 * The observations are given in the VEHICLE'S coordinate system.
	 * Your particles are located according to the MAP'S coordinate system.
	 * You will need to transform between the two systems.
	 * Keep in mind that this transformation requires both rotation AND translation (but no scaling).

	 for each particle

	    for each landmark
	        if landmark within the range
	            add it to landmark_within_range

        for each observation
            transform to map coordinates

	    findout the closest landmark (neighouest neighbour) - call dataAssociation

	    for each observation
	        update the weight..

	 */


	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	double x_denom = 2.0 * pow(std_x, 2);
	double y_denom = 2.0 * pow(std_y, 2);
	double dist_mult = 1.0 / (2.0*M_PI*std_x*std_y);

	// for each particle;
	for (unsigned int p = 0; p < num_particles; p++){

		// get particles properties

		double px = particles[p].x;
		double py = particles[p].y;
		double ptheta = particles[p].theta;

		// chose only landmarks within sensor range
        std::vector<LandmarkObs> landmarks_in_range = getLandMarksInRange (sensor_range, map_landmarks, px, py);

        // transform vehicle coordinate observations into map's coordinate
        std::vector<LandmarkObs> observations_m = transformToMapSpace (observations, px, py, ptheta);

        // perform data association
		dataAssociation(landmarks_in_range, observations_m);

		// update particle weight
        calculateWeight (p, x_denom, y_denom, dist_mult, landmarks_in_range, observations_m);

    }

}

void ParticleFilter::calculateWeight(unsigned int p,double x_denom, double y_denom, double dist_mult,
                                     const std::vector<LandmarkObs> &landmarks_in_range,
                                     const std::vector<LandmarkObs> &observations_m)  {
    particles[p].weight = 1.0;

    for (unsigned int i = 0; i < observations_m.size(); i++){

			int obs_index = observations_m[i].id;

			double o_x, o_y, obs_x, obs_y;
			o_x = observations_m[i].x;
			o_y = observations_m[i].y;

			double var_x = (landmarks_in_range[obs_index].x - o_x);
			double var_y = (landmarks_in_range[obs_index].y - o_y);
			double exponent = pow(var_x, 2) / x_denom + pow(var_y, 2) / y_denom;

			particles[p].weight *= dist_mult * exp(-exponent);
		}
}



std::vector<LandmarkObs>
ParticleFilter::transformToMapSpace(const std::vector<LandmarkObs> &observations, double px, double py, double ptheta)  {
    std::__1::vector<LandmarkObs> observations_m;
    for (unsigned int ob = 0; ob < observations.size(); ob++){

			double ox = observations[ob].x;
			double oy = observations[ob].y;

			double x = ox * cos(ptheta) - oy * sin(ptheta) + px;
			double y = ox * sin(ptheta) + oy * cos(ptheta) + py;
			observations_m.push_back({ observations[ob].id, x, y });
		}
    return observations_m;
}



std::vector<LandmarkObs> ParticleFilter::getLandMarksInRange(double sensor_range, const Map &map_landmarks, double px,
                                                             double py)  {
    std::__1::vector<LandmarkObs> landmarks_in_range;
    int landmark_size = map_landmarks.landmark_list.size();

    for (unsigned int mk = 0; mk < landmark_size; mk++) {

			float lx = map_landmarks.landmark_list[mk].x_f;
			float ly = map_landmarks.landmark_list[mk].y_f;
			int l_id = map_landmarks.landmark_list[mk].id_i;

			// find landmark distance from particle
			double lndmrk_dist = dist(px, py, lx, ly);

			if (lndmrk_dist < sensor_range){
				landmarks_in_range.push_back({ l_id, lx, ly });
			}
		}
    return landmarks_in_range;
}

void ParticleFilter::resample() {


	std::random_device rd;
	std::mt19937 gen(rd());

	// get all of the current weights
	std::vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	// create distribution
	std::discrete_distribution<> d(weights.begin(), weights.end());

	// placeholder for resampled particles
	std::vector<Particle> resampled_particles;
	resampled_particles.resize(num_particles);

	// sample new particles
	for (int n = 0; n < num_particles; n++) {
		int new_index = d(gen);
		resampled_particles[n] = particles[new_index];
	}
	particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
