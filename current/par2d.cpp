// export OMP_NUM_THREADS=2; clang++-17 -pedantic -Wall -fopenmp -Wunknown-pragmas -fsanitize=address -lm -lstdc++ -std=c++20 -ffast-math -O3 -march=native driver_parallel.cpp -o par && ./par

#include <array>
#include <omp.h>
#include <iostream>
#include <vector>
#include <tuple>

static inline std::tuple<int, int, int> revert(int s, int N = 30)
{
    auto k = s % N;
    auto j = ((s - k) / N) % N;
    auto i = (s - k - j*N) / (N * N);
    return {i, j, k};
}

static inline int convert(const std::tuple<int, int, int> &t, int N=30)
{
    auto [x,y,z] = t;
    return z + N*(y + N*x);
}
void print_nn(const std::array<std::tuple<int, int, int>, 6> &nn, int i)
{
    std::cout << i << ": MY NEIGHBORS ARE left " << convert(nn[0]) << " right " << convert(nn[1])
              << " bottom " << convert(nn[2]) << " top " << convert(nn[3]) << " back " << convert(nn[4])
              << " front " << convert(nn[5]) << "\n";
}

int main(int argc, char **argv)
{

    const int N = 30;

    int t[N*N*N] = {-1};
    for (int i=0;i<N*N*N;++i)t[i]=-1;

    int i,j,k;
#pragma omp parallel for private(i,j,k) schedule(static)
    for(i=0;i<N;i+=1)
        for(j=0;j<N;j+=1)
            for(k=0;k<N;k+=1)
            {
                if ((i+j+k) % omp_get_num_threads() == omp_get_thread_num())
                {
                auto tup = std::make_tuple((i>0? i-1:(i+1)%N),(j>0?j-1:(j+1)%N),(k>0?k-1:(k+1)%N));
                
                int idx = convert(tup, N); 
                //int left, right, up, down, front, back;
                //left = (i-1+N) % N; right = (i+1) % N;
                //down = (j-1+N)%N; up = (j+1)%N;
                //back = (k-1+N)%N; front = (k+1)%N;

                int x,y,z; x=i;y=j;z=k;
                std::array<std::tuple<int, int, int>, 6> my_neighbors;
                if (t[idx + omp_get_thread_num()] != -1) continue; 
                // left (with PBC, inner), then right (with PBC, inner)
                if (x == 0)   my_neighbors[0]  = {N-1, y, z};
                if (x > 0)    my_neighbors[0]  = {x-1, y, z};
                if (x == N-1) my_neighbors[1]  = {0, y, z};
                if (x < N-1)  my_neighbors[1]  = {x+1, y, z};

                if (y == 0)    my_neighbors[2] = {x, N-1, z};
                if (y > 0)     my_neighbors[2] = {x, y-1, z};
                if (y == N-1)  my_neighbors[3] = {x, 0, z};
                if (y < N-1)   my_neighbors[3] = {x, y+1, z};

                if (z == 0)   my_neighbors[4]  = {x, y, N-1};
                if (z > 0)    my_neighbors[4]  = {x, y, z-1};
                if (z < N-1)  my_neighbors[5]  = {x, y, z+1};
                if (z == N-1) my_neighbors[5]  = {x, y, 0};
#pragma unroll(6)
                for(int jj=0;jj<6;++jj) t[convert(my_neighbors[jj])] = omp_get_thread_num(); 
                t[idx] = omp_get_thread_num();
                }
            }

    //for (int i=N * (10 + N);i< N * (20 + 2*N);++i) std::cout << t[i] << " ";
    std::cout << "FIRST\n"; 
    for (int i=0;i< N*N;++i) std::cout << ((i%(N*N)==0)? "\n" : "") <<  t[i] << " ";
    std::cout << "\nMID\n";
    for (int i=N*N*(N-N/2);i< N*N*(N-N/2 + 1);++i) std::cout << ((i%(N*N)==0)? "\n" : "") <<  t[i] << " ";
    std::cout << "\nLAST\n";
    for (int i=N*N*(N-1);i< N*N*N;++i) std::cout << ((i%(N*N)==0)? "\n" : "") <<  t[i] << " ";

    std::cout << "\nfinal check\n";
    for(int i=0;i<N*N;++i) std::cout << (bool)(t[i] & t[i + N*N*(N-1)]) << " ";

    // THIS HERE should give a nice 3d "checkerboard" that can tell us then if the parallelization strategy is actually
    // working
    std::cout << "\n";
}
