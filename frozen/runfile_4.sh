#!/bin/bash

mpirun -np 1 -x GOMP_CPU_AFFINITY="0,2"   to_single 2 12  \
     : -np 1 -x GOMP_CPU_AFFINITY="4,6"   to_single 2 12  \
     : -np 1 -x GOMP_CPU_AFFINITY="8,10"  to_single 2 12  \
     : -np 1 -x GOMP_CPU_AFFINITY="12,14" to_single 2 12
