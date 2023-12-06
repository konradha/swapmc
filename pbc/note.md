### Compilation for profiling

```bash
clang-15 -pg -lm -lstdc++ -std=c++17 -O3 -march=native -ffast-math lattice3d.cpp -o lat && ./lat .1 .75 100000
```


to profile
```bash
gprof lat gmon.out > profile.txt
```

with perf (may need root access)
```bash
perf record -g -- path-to-executable args
#create a nice picture using gprof2dot
perf script | c++filt | gprof2dot -f perf | dot -Tpng -o profile.png
```
