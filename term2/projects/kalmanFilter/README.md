# Extended Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)


## Directory structire

data			-> Contains data needed for testing
templatecode		-> Contains template code from Udacity
implementation		-> Contains implementation for the project

CMakeLists.txt		-> Contains build make instructions
Docs
Eigen
lessons			-> Contains lessons code base


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./extendedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.

    - eg. `./extendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`


