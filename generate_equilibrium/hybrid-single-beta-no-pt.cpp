/*
 * mpicxx -pedantic -ffast-math -march=native -O3 -Wall -fopenmp -Wunknown-pragmas  -lm -lstdc++ -std=c++17 hybrid-single-beta-no-pt.cpp -o to_hybrid
 * mpirun -np 2 -x OMP_NUM_THREADS=4 to_hybrid 12
 */

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

#include <mpi.h>

void dump_slice(const short *__restrict lattice, const int N, const int i) {
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k)
      std::cout << lattice[k + N * (j + i * N)] << ",";
    std::cout << "\n";
  }
}

void dump_lattice(const short *__restrict lattice, int N) {
  for (int i = 0; i < N; ++i) {
    dump_slice(lattice, N, i);
    std::cout << "\n";
  }
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

int sample(const short *__restrict a, const short &N, const int &count,
           const int &part_type) {
  std::random_device rd_sample;
  std::mt19937 gen_sample(__rdtsc());
  int res = -1;
  {
    std::uniform_int_distribution<> dis(0, N - 1);
    while (res == -1) {
      const auto idx = dis(gen_sample);
      if (a[idx] == part_type)
        res = idx;
    }
  }
  return res;
}

int build_lattice(short *grid, const int N, const int num_red,
                  const int num_blue) {
  for (int i = 0; i < N * N * N; ++i)
    grid[i] = 0; // all sites vacant in the beginning
  int curr_red, curr_blue;
  curr_red = curr_blue = 0;

  while (curr_red < num_red) {
    try {
      auto site = sample_vacant(grid, N * N * N);
      grid[site] = 1;
      curr_red++;
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }
  }

  while (curr_blue < num_blue) {
    try {
      auto site = sample_vacant(grid, N * N * N);
      grid[site] = 2;
      curr_blue++;
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }
  }
  return 0;
}

static inline void exchange(short *__restrict grid, const int &N,
                            const int &site, const int &to) {
  const auto tmp = grid[site];
  grid[site] = grid[to];
  grid[to] = tmp;
}

static inline std::tuple<int, int, int> revert(int s, int L = 30) {
  const auto k = s % L;
  const auto j = ((s - k) / L) % L;
  const auto i = (s - k - j * L) / (L * L);
  return {i, j, k};
}

std::array<int, 6> get_neighbors(const int &i, const int &j, const int &k,
                                 const int &L) {
  int ii, kk, jj;
  ii = (i == 0 ? L - 1 : i - 1);
  jj = (j == 0 ? L - 1 : j - 1);
  kk = (k == 0 ? L - 1 : k - 1);
  const int f = k + L * (j + (ii % L) * L);
  const int b = k + L * (j + ((i + 1) % L) * L);
  const int u = k + L * ((jj % L) + i * L);
  const int d = k + L * (((j + 1) % L) + i * L);
  const int l = (kk % L) + L * (j + i * L);
  const int r = ((k + 1) % L) + L * (j + i * L);
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

std::unordered_map<int, std::array<int, 6>> nn;

#define ONETWO(x) ((x & 1) || (x & 2))
static inline float local_energy(short *__restrict grid, const int &N,
                                 const int &i, const int &j, const int &k) {
  const auto site = k + N * (j + N * i);
  if (grid[site] == 0)
    return 0.;

  const float connection = grid[site] == 1 ? 3. : 5.;
  const auto [l, r, u, d, f, b] = nn[site];
  float current =
      static_cast<float>(ONETWO(grid[u]) + ONETWO(grid[d]) + ONETWO(grid[l]) +
                         ONETWO(grid[r]) + ONETWO(grid[f]) + ONETWO(grid[b]));
  current = current - connection;
  return current * current;
}

static inline const float nn_energy(short *__restrict lattice, const int N,
                                    const int &i, const int &j, const int &k) {
  const auto site = k + N * (j + N * i);
  if (lattice[site] == 0)
    return 0.;
  float res = local_energy(lattice, N, i, j, k);
  //#pragma unroll(6)
  for (const auto &e : nn[site]) {
    const auto [m, l, n] = revert(e, N);
    res += local_energy(lattice, N, m, l, n);
  }
  return res;
}


// single thread calls this function
void fully_nonlocal_sweep(short *__restrict lattice, const int L,
           std::vector<std::mt19937> &gens,
           std::vector<std::uniform_real_distribution<>> &unis,
           std::vector<std::uniform_int_distribution<>> &indices,
           const float beta)
{
  // lattice points already to own lattice per thread
  // ie. lattices_per_rank[tid]
  const auto tid = omp_get_thread_num();
  for(int i = 0; i < L * L * L; ++i)
  {
    // is this even better than taking modulo?
    const int site = indices[tid](gens[tid]); 
    const int mv   = indices[tid](gens[tid]);
    if (lattice[site] == lattice[mv]) continue;
    const auto [si, sj, sk] = revert(site, L);
    const auto [mi, mj, mk] = revert(mv, L);
    const float E1 = nn_energy(lattice, L, si, sj, sk) + nn_energy(lattice, L, mi, mj, mk) ;
    exchange(lattice, L, site, mv);
    const float E2 = nn_energy(lattice, L, si, sj, sk) + nn_energy(lattice, L, mi, mj, mk) ;
    const float dE = E2 - E1;
    const auto crit = (std::exp(-beta * dE) < 1. ? std::exp(-beta * dE) : 1.);
    if (unis[tid](gens[tid]) < crit) continue;
    exchange(lattice, L, site, mv);
   }
}

int get_num_threads(void) {
  int num_threads = 1;
#pragma omp parallel
  {
#pragma omp single
    num_threads = omp_get_num_threads();
  }
  return num_threads;
}

int energy(short *__restrict lattice, const int & L)
{
    int e = 0;
    #pragma omp parallel for collapse(3) reduction(+ : e)
    for(int i=0;i<L;++i)
        for(int j=0;j<L;++j)
            for(int k=0;k<L;++k)
                e += local_energy(lattice, L, i, j, k);
    return e;
}

void collect_and_print(const int & rk, const int & sz, const int & num_threads, short *__restrict all_configs, short *__restrict rank_lattices, const int & padded_N, const int & L)
{
  MPI_Gather(rank_lattices, padded_N, MPI_SHORT, all_configs, padded_N, MPI_SHORT, 0, MPI_COMM_WORLD);
  if (rk == 0)
  {
    for (int r = 0; r < sz; ++r)
      for(int t = 0; t < num_threads; ++t)
          std::cout << energy(all_configs + (r * num_threads + t), L) << " ";
    std::cout << "\n";
  }
}



void seed(int k) { srand(time(NULL) + k); }

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "run as ./bin L\n";
    return 1;
  }

  int rk, sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  // seed all ranks differently
  seed(rk);

  const auto arg_n = argv[1];
  const auto L = atoi(arg_n);

  const float rho = .75;
  
  const float rho1 = .4 * rho;
  const float rho2 = rho - rho1;
  const int numparts = static_cast<int>(rho * L * L * L);

  const int r = static_cast<int>(rho1 * L * L * L);
  const int b = numparts - r;

  assert(rho1 + rho2 == rho);

  omp_set_num_threads(4);
  ::nn = get_nn_list(L);

#pragma omp parallel
  assert(omp_get_num_threads() == 4);

  const int num_threads = 4;

  short * rank_lattices = nullptr;

  const unsigned int align = 64;
  const auto padded_N = (L * L * L + (align - 1)) & ~(align - 1);

  if (posix_memalign((void **)&rank_lattices, align, num_threads * padded_N * sizeof(short)) != 0)
    return 1;

  // first thread in rank builds all lattices for rank
  for(int t = 0; t < num_threads; ++t)
    build_lattice(rank_lattices + t * padded_N, L, r, b);

  

  short * all_configs = nullptr;
  if (rk == 0) all_configs = (short *) malloc(sizeof(short) * padded_N * sz * num_threads);

  std::vector<std::mt19937> thread_generators(num_threads);
  std::vector<std::uniform_real_distribution<>> unis(num_threads);
  std::vector<std::uniform_int_distribution<>> indices(num_threads);

#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
#pragma omp critical
    {
      thread_generators[tid].seed(std::random_device()() + tid);
    }
  }

  const int num_configs = num_threads * sz;
  float * betas = (float *) malloc(sizeof(float) * num_configs);
  const float beta_min = 2.;
  const float beta_max = 4.;
  const float d = (beta_max - beta_min) / num_configs;
  for(int i = 0; i < num_configs; ++i)
      betas[i] = beta_min + i * d;

  

#pragma omp parallel
  {
    // every thread allocates its own RNG
    const auto tid = omp_get_thread_num();
    const auto max_idx = L * L * L - 1;
    if (tid == 0)
      for (int t = 0; t < omp_get_num_threads(); ++t) {
        unis[t] = std::uniform_real_distribution<>(0., 1.);
        indices[t] = std::uniform_int_distribution<>(0, max_idx);
      }
  }
  
  collect_and_print(rk, sz, num_threads, all_configs, rank_lattices, padded_N,  L);


  int nsweeps = 1000000;

  int printer = 10;

  for (int i = 1; i <= nsweeps + 1; ++i)
  {
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      const float my_beta = betas[rk * num_threads + tid];
      short * my_lattice = rank_lattices + tid * padded_N; 
      fully_nonlocal_sweep(my_lattice,  L, thread_generators, unis, indices, my_beta);
    }

    if (i % printer == 0)
    {
      collect_and_print(rk, sz, num_threads, all_configs, rank_lattices, padded_N,  L); 
      printer += 10000;
    }
  }
  collect_and_print(rk, sz, num_threads, all_configs, rank_lattices, padded_N,  L);



  free(rank_lattices); free(betas);
  if (rk == 0) free(all_configs);
  MPI_Finalize();
  return 0;
}

