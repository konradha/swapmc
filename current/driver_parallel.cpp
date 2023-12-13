// export OMP_NUM_THREADS=2; clang++-17 -pedantic -Wall -fopenmp -Wunknown-pragmas -fsanitize=address -lm -lstdc++ -std=c++20 -ffast-math -O3 -march=native driver_parallel.cpp -o par && ./par

#include <array>
#include <omp.h>
#include <iostream>
#include <vector>
#include <tuple>


// This file is mostly an exercise in converting the indices correctly
// to the goal of being able to massively parallelize the main local sweep
//
// also: a) do sites need to be shuffled?
//       b) coin flip where to start parallel sweeps (we're introducing some bias here)?

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

//std::vector<std::tuple<int, int, int>> get_neighbors(const std::vector<std::tuple<int, int, int>> & grid, int site, int N = 30)
//{
//    std::vector<std::tuple<int, int, int>> nn; nn.reserve(6);
//    auto [x, y, z] = revert(site, N);
//
//    if (x == 0) nn.push_back({N-1, y, z});
//    if (x == N-1) nn.push_back({0, y, z});
//    if (z == 0) nn.push_back({x, y, N-1});
//    if (z == N-1) nn.push_back({x, y, 0});
//    if (y == 0) nn.push_back({x, N-1, z});
//    if (y == N-1) nn.push_back({x, 0, z});
//
//    // internal
//    
//    if (x > 0) nn.push_back(
//
//
//
//    /*
//    // wrap around, PBC
//    if (i == 0)
//      e->neighbors[0] = k + ny * (j + (nx - 1) * nx);
//    if (i == nx - 1)
//      e->neighbors[3] = k + ny * (j + 0 * nx);
//    if (j == 0)
//      e->neighbors[1] = k + ny * ((ny - 1) + i * nx);
//    if (j == ny - 1)
//      e->neighbors[4] = k + ny * (0 + i * nx);
//    if (k == 0)
//      e->neighbors[2] = (nz - 1) + ny * (j + i * nx);
//    if (k == nz - 1)
//      e->neighbors[5] = 0 + ny * (j + i * nx);
//
//    // all internal neighbors
//    if (i > 0)
//      e->neighbors[0] = k + ny * (j + (i - 1) * nx);
//    if (j > 0)
//      e->neighbors[1] = k + ny * (j - 1 + i * nx);
//    if (k > 0)
//      e->neighbors[2] = k - 1 + ny * (j + i * nx);
//    if (i < nx - 1)
//      e->neighbors[3] = k + ny * (j + (i + 1) * nx);
//    if (j < ny - 1)
//      e->neighbors[4] = k + ny * (j + 1 + i * nx);
//    if (k < nz - 1)
//      e->neighbors[5] = k + 1 + ny * (j + i * nx);
//    */
//
//
//    return nn; 
//}

void print_nn(const std::array<std::tuple<int, int, int>, 6> &nn, int i)
{
    std::cout << i << ": MY NEIGHBORS ARE left " << convert(nn[0]) << " right " << convert(nn[1])
              << " bottom " << convert(nn[2]) << " top " << convert(nn[3]) << " back " << convert(nn[4])
              << " front " << convert(nn[5]) << "\n";
}

int main(int argc, char **argv)
{
    //std::vector<std::tuple<int, int, int>> grid;
    //int N = 30;

    //for(int i=0;i<N;++i)
    //    for(int j=0;j<N;++j)
    //        for(int k=0;k<N;++k)
    //            grid.push_back({i,j,k});

    //int i,j,k;
    //i = 10;
    //j = 2;
    //k = 3;
    //auto [ii, jj, kk] = grid[k + N * (j + i*N)]; 
    //std::cout << i << "?=" << ii << "\n";
    //std::cout << j << "?=" << jj << "\n";
    //std::cout << k << "?=" << kk << "\n";

    //auto [iii, jjj, kkk] = revert(k + N * (j + i*N), N);
    //
    //std::cout << i << "?=" << iii << "\n";
    //std::cout << j << "?=" << jjj << "\n";
    //std::cout << k << "?=" << kkk << "\n";


    int N = 30;

    auto grid = (int *) calloc(sizeof(int), N * N * N);
    for (int i=0;i< N * N * N; ++i)
    {
        if (i %2 != 0) grid[i] = 1;
        else grid[i] = 2;
    }

    int t[20] = {0, 0};

    
#pragma omp parallel for
    for (int i=0;i< N * N * N; i+=3)
    {
        auto [x, y, z] = revert(i, N);        
        std::array<std::tuple<int, int, int>, 6> my_neighbors;
 
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
        t[omp_get_thread_num()] += 1;
        
        //print_nn(my_neighbors, omp_get_thread_num()); 
    }

#pragma omp parallel for
    for (int i=0;i<omp_get_num_threads();++i)
        std::cout << "thread " << i << "=" << t[i] << "\n";
}
