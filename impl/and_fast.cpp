// clang-15 -fsanitize=address -lm -lstdc++ -std=c++17 -O3 -march=native and_fast.cpp -o fast_and
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

void and_arrays(bool * __restrict r, bool * __restrict o, bool * __restrict n, int N)
{
    int k = 0;
    int l = 32;
    for (; k < N - (l-1); k += l)
    {
        __m256i v1 = _mm256_loadu_si256((__m256i*)&o[k]);
        __m256i v2 = _mm256_loadu_si256((__m256i*)&n[k]);
        __m256i v3 = _mm256_and_si256(v1, v2);
        _mm256_storeu_si256((__m256i*)&r[k], v3);
    }

    for(;k<N;++k)
        r[k] = o[k] & n[k];
}

int main(int argc, char **argv)
{
    if (argc < 2) return 1;
    auto arg = argv[1];
    auto N = (int)atof(arg); 
    
    auto padded_N = (N+31) & ~31;
    bool *o; bool *n; bool *r;
    if (posix_memalign((void**)&o, 32, padded_N * sizeof(bool)) != 0) return 1;
    if (posix_memalign((void**)&n, 32, padded_N * sizeof(bool)) != 0) return 1;
    if (posix_memalign((void**)&r, 32, padded_N * sizeof(bool)) != 0) return 1;
    

    for (int i=0;i<N;++i) if (i % 3 == 0) o[i] = true;
    for (int i=0;i<N;++i) if (i % 77 == 0) n[i] = true;


    auto t0 = __rdtsc(); 
    and_arrays(r, o, n, padded_N);
    auto tf = __rdtsc();

    std::cout << tf - t0 << "\n";
    free(o); free(n); free(r);
}
