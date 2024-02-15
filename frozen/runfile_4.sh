#!/bin/bash
L=20
mpirun -np 1 -x GOMP_CPU_AFFINITY="0,1,2,3"    to_single 4  ${L}  \
     : -np 1 -x GOMP_CPU_AFFINITY="4,5,6,7"    to_single 4  ${L}  \
     : -np 1 -x GOMP_CPU_AFFINITY="8,9,10,11"   to_single 4 ${L}  \
     : -np 1 -x GOMP_CPU_AFFINITY="12,13,14,15" to_single 4 ${L}
