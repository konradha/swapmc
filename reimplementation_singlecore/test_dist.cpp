#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>

#include <random>
#include <stdexcept>
#include <stdlib.h>

#include <immintrin.h>
#include <unordered_map>

#include <cmath>

#define IDX(i, j, k, L) k + L * ( j + i * L)


void print_slice(short * g, const int & i, const int & L)
{
    for(int j=0;j<L;++j)
    {
        for(int k=0;k<L;++k)
            std::cout << g[IDX(i,j,k,L)] << ",";
        std::cout << "\n";
    }

}

void print_grid(short *g, const int & L)
{
    for(int i=0;i<L;++i)
        print_slice(g, i, L);
}


std::array<int, 6> get_neighbors(const int &i, const int &j, const int &k,
                                 const int &N) {
  int ii, kk, jj;
  // different to Python: no automatic wrap-around for negative values
  ii = (i==0? N-1 : i-1);
  jj = (j==0? N-1 : j-1);
  kk = (k==0? N-1 : k-1);
  const int f = k + N * ( j + (ii % N)* N);
  const int b = k + N * ( j + ((i+1) % N)* N);
  const int u = k + N * ((jj % N) + i* N);
  const int d = k + N * (((j+1) % N) + i* N);
  const int l = (kk % N) + N*(j + i *N);
  const int r = ((k+1) % N) + N*(j + i *N);
  return {u, d, l, r, f, b};
}

std::unordered_map<int, std::array<int, 6>> get_nn_list(const int &N) {
  std::unordered_map<int, std::array<int, 6>> neighbor_map;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k) {
        const auto site = k + N * (j + i * N);
        neighbor_map[site] = get_neighbors(i, j, k, N);
      }
  return neighbor_map;
}

static inline void exchange(short *__restrict grid, const int &N,
                            const int &site, const int &to) {
  const auto tmp = grid[site];
  grid[site] = grid[to];
  grid[to] = tmp;
}

static inline std::tuple<int, int, int> revert(int s, int N = 30) {
  const auto k = s % N;
  const auto j = ((s - k) / N) % N;
  const auto i = (s - k - j * N) / (N * N);
  return {i, j, k};
}



std::random_device rd_sample;
std::mt19937 gen_sample(__rdtsc());
int sample_vacant(short *grid, int sz) {
  std::uniform_int_distribution<> dis(0, sz - 1);
  // our algorithm assumes that the number of vacant spots is always larger than
  // num_reds + num_blue
  for (;;) {
    const int rand = dis(gen_sample);
    if (grid[rand] == 0)
      return rand;
  }
}

//int build_lattice(short *grid, const int N, const int num_red,
//                  const int num_blue) {
//  for (int i = 0; i < N * N * N; ++i)
//    grid[i] = 0; // all sites vacant in the beginning
//  int curr_red, curr_blue;
//  curr_red = curr_blue = 0;
//
//  while (curr_red < num_red) {
//    try {
//      auto site = sample_vacant(grid, N * N * N);
//      grid[site] = 1;
//      curr_red++;
//    } catch (const std::exception &e) {
//      std::cerr << e.what() << std::endl;
//      return 1;
//    }
//  }
//
//  while (curr_blue < num_blue) {
//    try {
//      auto site = sample_vacant(grid, N * N * N);
//      grid[site] = 2;
//      curr_blue++;
//    } catch (const std::exception &e) {
//      std::cerr << e.what() << std::endl;
//      return 1;
//    }
//  }
//  return 0;
//}

int build_lattice(short* grid, const int N, const int num_red, const int num_blue) {
    std::random_device rd;
    std::mt19937 gen_red(rd());
    std::mt19937 gen_blue(rd());

    std::uniform_int_distribution<int> dist(0, N*N*N-1);

    int curr_red = 0;
    int curr_blue = 0;
    int num_vacant = N*N*N;
   for (int i = 0; i < N * N * N; ++i)
    grid[i] = 0; // all sites vacant in the beginning

    while (curr_red < num_red && num_vacant > 0) {
        int site = dist(gen_red);
        if (grid[site] == 0) {
            grid[site] = 1;
            curr_red++;
            num_vacant--;
        }
    }

    while (curr_blue < num_blue && num_vacant > 0) {
        int site = dist(gen_blue);
        if (grid[site] == 0) {
            grid[site] = 2;
            curr_blue++;
            num_vacant--;
        }
    }

    return 0;
}



static inline int count_nn(short *lattice, const int & pos, std::unordered_map<int, std::array<int, 6>> & nn)
{
    int res = 0;
    auto nnn = nn[pos];
    for(const auto & n : nnn)
        res += static_cast<int>(lattice[n] > 0 && lattice[n] < 3);
    return res;
}


float local_e(short * lattice, const int & pos, const int & L, std::unordered_map<int, std::array<int, 6>> & nn)
{
    if (lattice[pos] == 0) return 0.;
    int pref = 0;
    if (lattice[pos] == 1) pref = 3;
    else if (lattice[pos] == 2) pref = 5;
    const auto num_neighbors = static_cast<float>(count_nn(lattice, pos, nn));
    return (num_neighbors - static_cast<float>(pref)) * (num_neighbors - static_cast<float>(pref)); 
}


float summed_local_energy(short * lattice, const int & s, const int & L, std::unordered_map<int, std::array<int, 6>> & nn)
{
    float curr = local_e(lattice, s, L, nn);
    const auto nnn = nn[s];
    for (const auto & n : nnn)
        curr += local_e(lattice, n, L, nn);
    return curr;
}




void iterate_energies(short *g, const int & L, std::unordered_map<int, std::array<int, 6>> & nn)
{
    for(int i=0;i<L;++i)
    {
        for(int j=0;j<L;++j)
        {
            for(int k=0;k<L;++k)
            {
                // for each site, compute energy differences when moving all surroundings
                // (if site contains a particle)
                const auto site = IDX(i,j,k,L);
                if (g[site] == 0) continue;
                const auto nnn = nn[site];
                for(const auto n : nnn)
                {
                    if (g[n] != 0) continue; // move to empty slots only
                    const float E1 = summed_local_energy(g, site, L, nn) + summed_local_energy(g, n, L, nn);
                    exchange(g, L, site, n);
                    const float E2 = summed_local_energy(g, site, L, nn) + summed_local_energy(g, n, L, nn);
                    exchange(g, L, site, n); 
                    std::cout << E2 - E1 << "\n";
                }
            }
        }
    }
}






int main()
{
    const int L = 12;
    short * grid = (short *) malloc( sizeof(short) * L * L * L);
    const float rho = .75;
    const float rho1 = .6 * rho;
    const float rho2 =  1. - rho;

    const float floatN = rho * L * L * L;
    const int precision = std::numeric_limits<int>::max_digits10;
    const int N = static_cast<int>(std::round(floatN * std::pow(10, precision))) / std::pow(10, precision);

    const float floatR = rho1 * L * L * L;
    const int r = static_cast<int>(std::round(floatR * std::pow(10, precision))) / std::pow(10, precision);

    const int b = N - r;
    std::cout << N << " " << r << " " << b << "\n";
    
    //build_lattice(grid, L, r, b);
    //auto neighbor_index_list = get_nn_list(L);
    //iterate_energies(grid, L, neighbor_index_list);
    ////print_grid(grid, L);
    free(grid);
}
