#!/bin/bash

# mpicxx -pg -pedantic -ffast-math -march=native -O3 -Wall -fopenmp -Wunknown-pragmas  -lm -lstdc++ -std=c++17 omp_overlap.cpp -o overla

#mpirun -np 1 -x GOMP_CPU_AFFINITY="0,2,4,6" ./ptemper $1 4 \
#        : -np 1 -x GOMP_CPU_AFFINITY="8,10,12,14" ./ptemper $1 4

mpirun -np 1 -x GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./ptemper $1 8
