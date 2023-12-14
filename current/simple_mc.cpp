#include <algorithm>
#include <cassert>
#include <omp.h>
#include <iostream>

#include <stdlib.h>
#include <random>
#include <stdexcept>


// can do along axis
void print_slice(int *lattice, int N, int i)
{
    for(int j=0;j<N;++j)
    {
        for(int k=0;k<N;++k)
            std::cout << lattice[k + N * (j + i*N)] << " ";
        std::cout << "\n";
    }
}


std::random_device rd_sample;
std::mt19937 gen_sample(__rdtsc());
int sample_vacant(int *grid, int sz) {
  std::uniform_int_distribution<> dis(0, sz - 1);
  // our algorithm assumes that the number of vacant spots is always larger than
  // num_reds + num_blue
  for(;;)
  {
        int rand = dis(gen_sample);
        if (grid[rand] == 0) return rand;
  }
}


int build_lattice(int *grid, const int N, const int num_red, const int num_blue)
{
    for (int i=0;i<N*N*N;++i) grid[i] = 0; // all sites vacant in the beginning
    int curr_red, curr_blue; curr_red = curr_blue = 0;

    while (curr_red < num_red)
    {
        try {
            auto site  = sample_vacant(grid, N*N*N);
            grid[site] = 1; 
            curr_red++;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return 1;
        }
    }

    while (curr_blue < num_blue)
    {
        try {
            auto site  = sample_vacant(grid, N*N*N);
            grid[site] = 2; 
            curr_blue++;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return 1;
        }
    } 
    return 0;
}


static inline void exchange(int *grid, const int N, const int site, const int to)
{
    //std::swap(&grid[site], &grid[to]); <- redo
    auto tmp = grid[site]; grid[site] = grid[to];
    grid[to] = tmp;
}

static inline float local_energy(int *grid, const int N, const int i, const int j, const int k)
{
    auto site = k + N * (j + N * i);    
    if (grid[site] == 0) return 0.;

    float connection = (float)(grid[site] == 1? 3 : 5);
    float current = 0.;
    int l,r,u,d,f,b; // left, right, up, down, front, back
    if ( i == 0  ) {u = k + N * (j + (N-1) * N);}
    if ( i == N-1) {d = k + N *  j            ;}
    if ( j == 0  ) {l = k + N * ((N-1) + i * N);}
    if ( j == N-1) {r = k + N * (i * N);}
    if ( k == 0  ) {f = N-1 + N*(j + i*N);}
    if ( k == N-1) {b = N*(j + i * N);}
    if ( i > 0 )   {u = k + N * (j + (i-1) * N);}
    if ( i <N-1)   {d = k + N * (j + (i+1) * N);}
    if ( j > 0 )   {l = k + N * (j-1 + i*N);}
    if ( j <N-1)   {r = k + N * (j+1 + i*N);}
    if ( k > 0 )   {f = k-1 + N * (j + i*N);}
    if ( k <N-1)   {b = k+1 + N * (j + i*N);}

    for (const auto & nn : {u,d,l,r,f,b})
    {
        if (!grid[nn]) continue;
        current += 1.;
    }
    current = connection - current;
    return current * current;

}


void sweep(int *lattice, const int N, const float beta=1.)
{
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 5);
    std::uniform_real_distribution<> uni(0., 1.);
 

    int ii,jj,kk; ii=jj=kk=0;
     
    for (ii=0;ii<3;++ii)
    #pragma omp parallel for collapse(1) firstprivate(gen, dis, uni) 
                for(int i=ii;i<N-3+1;i+=3)
                { 
                    for(int j=jj;j<N-3+1;++j)
                        for(int k=kk;k<N-3+1;++k)
                        {
     
                               

                            int l,r,u,d,f,b; // left, right, up, down, front, back
                                             // TODO check if all of these work // if there's missing
                            l = r = u = d = f = b = 0; 
                            if ( i == 0  ) {u = k + N * (j + (N-1) * N);}
                            if ( i == N-1) {d = k + N *  j            ;}
                            if ( j == 0  ) {l = k + N * ((N-1) + i * N);}
                            if ( j == N-1) {r = k + N * (i * N);}
                            if ( k == 0  ) {f = N-1 + N*(j + i*N);}
                            if ( k == N-1) {b = N*(j + i * N);}

                            if ( i > 0 )   {u = k + N * (j + (i-1) * N);}
                            if ( i <N-1)   {d = k + N * (j + (i+1) * N);}
                            if ( j > 0 )   {l = k + N * (j-1 + i*N);}
                            if ( j <N-1)   {r = k + N * (j+1 + i*N);}
                            if ( k > 0 )   {f = k-1 + N * (j + i*N);}
                            if ( k <N-1)   {b = k+1 + N * (j + i*N);}

                            
                            assert(l >= 0 && l < N*N*N && "it's l");
                            assert(r >= 0 && r < N*N*N && "it's r");
                            assert(u >= 0 && u < N*N*N && "it's u");
                            assert(d >= 0 && d < N*N*N && "it's d");
                            assert(f >= 0 && f < N*N*N && "it's f");
                            assert(b >= 0 && b < N*N*N && "it's b");

                            float E1 = local_energy(lattice, N, i, j, k);
                            int nn[6] = {u,d,l,r,f,b};
                            auto mv = nn[dis(gen)];
                            exchange(lattice, N, k + N*(j + i*N), mv);
                            float E2 = local_energy(lattice, N, i, j, k);
                            float dE = E2 - E1; 
                            if (uni(gen) < std::exp(-beta * dE)) continue;  
                            exchange(lattice, N, mv, k + N*(j + i*N)); 
                        }
                }
}

void sweep_single(int *lattice, const int N, const float beta=1.)
{
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 5);
    std::uniform_real_distribution<> uni(0., 1.);
 

    int ii,jj,kk; ii=jj=kk=0;
     
     
    for(int i=0;i<N;++i)
    { 
        for(int j=jj;j<N;++j)
            for(int k=kk;k<N;++k)
            { 
                   

                int l,r,u,d,f,b; // left, right, up, down, front, back
                                 // TODO check if all of these work // if there's missing
                l = r = u = d = f = b = 0; 
                if ( i == 0  ) {u = k + N * (j + (N-1) * N);}
                if ( i == N-1) {d = k + N *  j            ;}
                if ( j == 0  ) {l = k + N * ((N-1) + i * N);}
                if ( j == N-1) {r = k + N * (i * N);}
                if ( k == 0  ) {f = N-1 + N*(j + i*N);}
                if ( k == N-1) {b = N*(j + i * N);}

                if ( i > 0 )   {u = k + N * (j + (i-1) * N);}
                if ( i <N-1)   {d = k + N * (j + (i+1) * N);}
                if ( j > 0 )   {l = k + N * (j-1 + i*N);}
                if ( j <N-1)   {r = k + N * (j+1 + i*N);}
                if ( k > 0 )   {f = k-1 + N * (j + i*N);}
                if ( k <N-1)   {b = k+1 + N * (j + i*N);}

                
                assert(l >= 0 && l < N*N*N && "it's l");
                assert(r >= 0 && r < N*N*N && "it's r");
                assert(u >= 0 && u < N*N*N && "it's u");
                assert(d >= 0 && d < N*N*N && "it's d");
                assert(f >= 0 && f < N*N*N && "it's f");
                assert(b >= 0 && b < N*N*N && "it's b");

                float E1 = local_energy(lattice, N, i, j, k);
                int nn[6] = {u,d,l,r,f,b};
                auto mv = nn[dis(gen)];
                exchange(lattice, N, k + N*(j + i*N), mv);
                float E2 = local_energy(lattice, N, i, j, k);
                float dE = E2 - E1; 
                if (uni(gen) < std::exp(-beta * dE)) continue;  
                exchange(lattice, N, mv, k + N*(j + i*N)); 
            }
    }
}

int main()
{
    int N=30;
    int r=1000; int b=100;

    int *lattice = nullptr;
    const unsigned int align = 64;
    auto padded_N = (N*N*N + (align-1)) & ~(align-1);
    if (posix_memalign((void**)&lattice, align, padded_N * sizeof(int)) != 0) return 1;
    build_lattice(lattice, N, r, b);
    
    std::cout << "scale in 1e6 epochs\n";
    int nsteps = 1000;
    float singles = 0.;
    for (int i=0;i<nsteps;++i)
    {
        auto t0 = __rdtsc(); 
        sweep_single(lattice, N);
        auto tf = __rdtsc();
        singles += tf-t0;
    }
    std::cout << "single core implementation: " << singles / nsteps / 1e6 << "\n";

    for (int i=1;i<9;++i)
    {
        float curr = 0.;
        omp_set_num_threads(i);        
        for (int i=0;i<nsteps;++i)
        {
            auto t0 = __rdtsc();
            sweep(lattice, N);
            auto tf = __rdtsc();
            curr += tf - t0;
        }
        std::cout << "OMP with " << i << " cores: " << curr / nsteps / 1e6 << "\n";
    }
    free(lattice);
    
}
