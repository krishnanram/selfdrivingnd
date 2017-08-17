# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---



## Basic Build Run Instructions

1. Make a build directory: `mkdir build && cd build`
2. Compile: `cmake .. && make`
3. Run it
    ./pid $p $i $d $speed_goal $tp $ti $d $twiddle_flag


Alternatively

1. cd to ./implementation directory
2. execute ./run.sh
    * run.sh will compile and run the program with following input

        ./pid $p $i $d $speed_goal $tp $ti $d $twiddle_flag

        where

        p=0.27
        i=0.004
        d=3
        speed_goal=30.0
        tp=0.1
        ti=0.002
        td=12
        twiddle_flag=true



## How to run with Simulator

    1) Start the Udacity simulator
    2) Run the PID controller code as explained in the "Build and Run instructions" section


## Output

    Observe how the car is able to navigate the road. Increase the speed and try again. Adjust the kd, ki params if needed