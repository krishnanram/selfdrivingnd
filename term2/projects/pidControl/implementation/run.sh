#!/bin/bash

echo "compiling ..."
cd /opt/github/public/selfdrivingnd/term2/projects/pidControl/build
make

cd /opt/github/public/selfdrivingnd/term2/projects/pidControl/implementation
echo "running ..."

p=0.27
i=0.004
d=3
speed_goal=30.0
tp=0.1
ti=0.002
td=12

twiddle_flag=true

echo "p=" $p 
echo "i=" $i
echo "d=" $d
echo "speed_goal=" $speed_goal
echo "tp=" $tp
echo "ti=" $ti
echo "td=" $td
echo "twiddle_flag=" $twiddle_flag

/opt/github/public/selfdrivingnd/term2/projects/pidControl/build/pid $p $i $d $speed_goal $tp $ti $d $twiddle_flag
