#!/bin/bash

# ALSO: this script needs an argument to run
mpirun -np 1 -x OMP_NUM_THREADS=8 -x GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./to_mpi $1
