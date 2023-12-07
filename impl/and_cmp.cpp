// clang-15 -Wunknown-pragmas -lm -lstdc++ -std=c++17 -O3 -march=native and_cmp.cpp -o cmp_and
#include <stdlib.h>
#include <iostream>
#include <immintrin.h>

#include <random>

void sample(bool* a, int size){
    std::random_device rd;
    std::mt19937 gen(__rdtsc());
    std::uniform_int_distribution<> dis(0, 1);

    int rand = dis(gen);
    for (int i = 0; i < size; ++i)
        if (dis(gen)) a[i] = 1;
}


void and_arrays(bool * __restrict r, bool * __restrict o, bool * __restrict n, int N)
{
    int k = 0;
    for(;k<N;++k)
        r[k] = o[k] & n[k];
}

void and_arrays_intrin(bool * __restrict r, bool * __restrict o, bool * __restrict n, int N)
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

void and_arrays_intrin_cc(bool * __restrict r, bool * __restrict o, bool * __restrict n, int N)
{
    int k = 0;
    int l = 32;
#pragma unroll(8)
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
    if (argc != 3)
    {
        std::cout << "use: ./exe N align\n";
        return 1;
    }
    
    auto arg_N = argv[1];
    auto arg_a = argv[2];
    auto N = (int)atof(arg_N);
    int align = (int)atof(arg_a);

    auto padded_N = (N+(align-1)) & ~(align-1);
    std::cout << "PADDED N: " << padded_N << "\n";
    bool *o; bool *n; bool *r;
    if (posix_memalign((void**)&o, align, padded_N * sizeof(bool)) != 0) return 1;
    if (posix_memalign((void**)&n, align, padded_N * sizeof(bool)) != 0) return 1;
    if (posix_memalign((void**)&r, align, padded_N * sizeof(bool)) != 0) return 1;

    for(int k=0;k<64;++k)
    {
        sample(o, padded_N);
        sample(n, padded_N);
        auto t0 = __rdtsc();
        and_arrays(r, o, n, N);
        auto tf = __rdtsc();

        sample(o, padded_N);
        sample(n, padded_N);
        auto t0_intrin = __rdtsc();
        and_arrays_intrin(r, o, n, N);
        auto tf_intrin = __rdtsc();

        sample(o, padded_N);
        sample(n, padded_N);
        auto t0_intrin_cc = __rdtsc();
        and_arrays_intrin_cc(r, o, n, N);
        auto tf_intrin_cc = __rdtsc();

        std::cout <<  tf - t0 <<","<< tf_intrin - t0_intrin << "," << tf_intrin_cc - t0_intrin_cc << "\n";  
    }

    //std::cout << "usual:  " << tf - t0 << "\n";
    //std::cout << "intrin: " << tf_intrin - t0_intrin << "\n";
    

    free(o); free(n); free(r);
}

