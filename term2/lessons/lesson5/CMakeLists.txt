project(lesson5)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(sources1
        main_chapter7.cpp
    )

set(sources2

        laser-measurement/main.h
        laser-measurement/measurement_package.h
        laser-measurement/KalmanFilter.h
        laser-measurement/Tracking.h

        laser-measurement/main.cpp
        laser-measurement/KalmanFilter.cpp
        laser-measurement/Tracking.cpp

        )

add_executable(lesson5 ${sources1})
add_executable(lesson51 ${sources2})