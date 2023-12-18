#!/bin/bash

for b in .1 .5 .8 1. 1.5 2. 2.5 3.
do bash /home/konrad/code/rsmi/swapmc/gblas/current/start_hybrid.sh $b > /home/konrad/code/rsmi/swapmc/gblas/current/simpledata_$b.txt 
done
