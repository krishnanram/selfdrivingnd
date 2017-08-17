#!/bin/sh

echo "compiling ..."
g++ main.cpp tools.cpp ukf.cpp -o run.out

chmod +x ./run.out
echo "running ..."
./run.out ../data-kalman/obj_pose-laser-radar-synthetic-input.txt ./out.txt
