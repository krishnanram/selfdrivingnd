#ifndef PID_H
#define PID_H

class PID {
public:

  double current_cte ;
  double integral_cte ;
  double difference_cte ;
  double previous_cte ;

  double sum_square_error ;
  double sum_absolute_value_error ;
  double mean_squared_error ;
  double twiddle_best_error ;

  int counter ;


  double Kp ;
  double Ki ;
  double Kd ;

  double tune_Kp ;
  double tune_Ki ;
  double tune_Kd ;
  int twiddle_counter ;

  float speed_goal ;

  double sum_speed ;
  double mean_speed ;
  double previous_steer_value ; 
  double max_speed ;


  PID();


  virtual ~PID();


  void Init(double Kp, double Ki, double Kd);


  void UpdateError(double cte);


  double TotalError();


};

#endif /* PID_H */
