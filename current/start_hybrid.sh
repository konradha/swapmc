#!/bin/bash
# this one here seems to work (on my topology, AMD Ryzen ...)
#mpirun -np 1 -x OMP_NUM_THREADS=4 -x GOMP_CPU_AFFINITY="0,2,4,6" ./to_mpi : -np 1 -x OMP_NUM_THREADS=4 -x GOMP_CPU_AFFINITY="8,10,12,14" ./to_mpi


mpirun -np 1 -x OMP_NUM_THREADS=2 -x GOMP_CPU_AFFINITY="0,2" ./to_mpi $1 \
        : -np 1 -x OMP_NUM_THREADS=2 -x GOMP_CPU_AFFINITY="4,6" ./to_mpi $1 \
        : -np 1 -x OMP_NUM_THREADS=2 -x GOMP_CPU_AFFINITY="8,10" ./to_mpi $1 \
        : -np 1 -x OMP_NUM_THREADS=2 -x GOMP_CPU_AFFINITY="12,14" ./to_mpi $1

