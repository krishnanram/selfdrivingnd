#!/bin/sh

echo "compiling ..."
g++ main.cpp tools.cpp ukf.cpp -o run.out

chmod +x ./run.out
echo "running ..."
./run.out
