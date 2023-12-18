#!/bin/bash

# enable local testing of hybrid openMP + MPI program

taskset -c 0-3 ./to_mpi \
       -x OMP_NUM_THREADS=4 \
       -x GOMP_CPU_AFFINITY="0-3" 
       
taskset -c 4-7 ./to_mpi \
       -x OMP_NUM_THREADS=4 \
       -x GOMP_CPU_AFFINITY="4-7"  

mpirun -np 2
