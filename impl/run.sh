#!/bin/bash

TARGET="./lat"
SWEEPS=1000000000

for i in {1..75}
do
    for ro in {8..16} #$(seq 0.4 0.05 0.8)
    do
        # parallel
        invtmp=$(echo "scale=1; $i/10" | bc)
        rho=$(echo "scale=2; $ro/20" | bc) 
        ${TARGET} ${invtmp} ${rho} ${SWEEPS} > data/epp_${invtmp}_${rho}.txt &
    done
    
    wait # ! running with `&`
done
