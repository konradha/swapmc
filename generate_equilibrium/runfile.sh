#!/bin/bash

mpirun -np 1 -x GOMP_CPU_AFFINITY="0,2,4,6"    to_hybrid 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="8,10,12,14" to_hybrid 12   
