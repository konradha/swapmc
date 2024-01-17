#!/bin/bash

EXEC=checkerboard


for s in 0 1; do
    for L in 8 12 20; do
        echo "running system size L=${L}, swap enabled=${s}"
        for i in 1 2 3 4 5; do 
            for d in 0; do
                ./${EXEC} ${i}.${d} 8 $L $s > run_L_${L}_beta_${i}.${d}_sliced_${s}.txt
            done
        done
    done
done
