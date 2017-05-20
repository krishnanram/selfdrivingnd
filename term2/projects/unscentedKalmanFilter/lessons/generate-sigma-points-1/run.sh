#!/bin/sh

echo "compiling ..."
g++ main.cpp ukf.cpp -o run.out

chmod +x ./run.out
echo "running ..."
./run.out
