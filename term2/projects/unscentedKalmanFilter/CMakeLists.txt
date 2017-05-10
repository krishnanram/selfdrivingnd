project(unscentedKalmanFilter)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(implementation

		implementation/ukf.cpp
		implementation/main.cpp
		implementation/tools.cpp
		)


#add_executable(extendedKalmanFilterTempl ${extendedSrc})
add_executable(unscentedKalmanFilter ${implementation})

