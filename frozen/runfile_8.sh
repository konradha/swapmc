#!/bin/bash

mpirun -np 1 -x GOMP_CPU_AFFINITY="0,1"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="2,3"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="4,5"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="6,7"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="8,9"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="10,11"  to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="12,13" to_single 2 12    \
     : -np 1 -x GOMP_CPU_AFFINITY="14,15" to_single 2 12
