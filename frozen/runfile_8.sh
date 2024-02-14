#!/bin/bash

mpirun -np 1 -x GOMP_CPU_AFFINITY="0"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="2"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="4"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="6"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="8"   to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="10"  to_single 2 12   \
     : -np 1 -x GOMP_CPU_AFFINITY="12" to_single 2 12    \
     : -np 1 -x GOMP_CPU_AFFINITY="14" to_single 2 12
