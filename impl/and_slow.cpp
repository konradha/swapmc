#include <stdlib.h>
#include <iostream>

void and_arrays(bool * __restrict r, bool * __restrict o, bool * __restrict n, int N)
{
    int k = 0;
    for(;k<N;++k)
        r[k] = o[k] & n[k];
}


int main(int argc, char **argv)
{
    if (argc < 2) return 1;
    auto arg = argv[1];
    auto N = (int)atof(arg);
    int align = 64;
    auto padded_N = (N+(align-1)) & ~(align-1);
    bool *o; bool *n; bool *r;
    if (posix_memalign((void**)&o, align, padded_N * sizeof(bool)) != 0) return 1;
    if (posix_memalign((void**)&n, align, padded_N * sizeof(bool)) != 0) return 1;
    if (posix_memalign((void**)&r, align, padded_N * sizeof(bool)) != 0) return 1;


    for (int i=0;i<N;++i) if (i % 3 == 0) o[i] = true;
    for (int i=0;i<N;++i) if (i % 77 == 0) n[i] = true;
    auto t0 = __rdtsc();
    and_arrays(r, o, n, N);
    auto tf = __rdtsc();

    std::cout << tf - t0 << "\n";
    free(o); free(n); free(r);
}
