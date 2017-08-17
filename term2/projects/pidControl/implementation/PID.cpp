#include "PID.h"
#include <math.h>
#include <iostream>

using namespace std;


PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {

  previous_cte      = 0.0 ;
  integral_cte      = 0.0 ;
  difference_cte    = 0.0 ;
  sum_square_error  = 0.0 ;
  counter			 = 0 ;

  sum_absolute_value_error = 0.0 ;
  twiddle_best_error = 999999.0 ;

  PID::Kp = Kp ;
  PID::Ki = Ki ;
  PID::Kd = Kd ;

  tune_Kp = .1 ;
  tune_Ki = 2.76871e-46 ;
  tune_Kd = 8.1425e-38  ;
  twiddle_counter = 0 ;

  speed_goal = 0.0 ;

  sum_speed = 0.0 ;
  mean_speed = 0.0 ;
  previous_steer_value = 0.0 ;
  max_speed = 0.0 ;



}

void PID::UpdateError(double cte) {

	counter += 1 ;

	current_cte = cte ;

	difference_cte = cte - previous_cte ;

    integral_cte += cte ;

	if ( counter % 1000 == 0) {
		integral_cte = integral_cte / 2 ;
	}

	sum_absolute_value_error += fabs(cte) ;
	sum_square_error += cte * cte ;
	mean_squared_error = sum_square_error / counter ;

	// debugging
	if (counter % 100 == 0) {
		
		cout << "\t" << endl ;

		cout << "Counter\t" << counter <<
		"  current_cte_\t" << current_cte <<
		"  previous_cte\t" << previous_cte <<
		"  sum_absolute_value_error_\t" << sum_absolute_value_error <<
		"  mean_squared_error_\t" << mean_squared_error << endl ;
	}

	previous_cte = cte ;

}

double PID::TotalError() {

	if (counter % 100 == 0) {

		cout << "P total error: " << - Kp * current_cte << endl ;
        cout << "I total error: " << - Kd * difference_cte << endl ;
        cout << "D total error: " << - Ki * integral_cte << endl ;

	}
	
	return  - Kp * current_cte - Kd * difference_cte - Ki * integral_cte ;
}

