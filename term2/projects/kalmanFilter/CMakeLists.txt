project(extendedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(sources

        implementation/FusionEKF.cpp
        implementation/kalman_filter.cpp
        implementation/main.cpp
        implementation/tools.cpp)


add_executable(extendedKF ${sources})
