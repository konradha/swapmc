#!/bin/bash

# ALSO: this script needs an argument to run
# the best one:
#mpirun -np 1 -x OMP_NUM_THREADS=8 -x GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./to_mpi $1

# trying to max it out
mpirun -np 1 -x OMP_NUM_THREADS=10 -x GOMP_CPU_AFFINITY="0,1,2,3,4,6,8,10,12,14" ./to_mpi $1
