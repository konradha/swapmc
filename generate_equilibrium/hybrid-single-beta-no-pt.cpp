#include "maps.h"

#include <algorithm>
#include <cassert>
#include <omp.h>
#include <random>
#include <stdexcept>

#include <cmath>
#include <mpi.h>

int logand(int *__restrict lat1, int *__restrict lat2, const int &L) {
  int x = 0;
  for (int i = 0; i < L * L * L; ++i)
    x += (int)(lat1[i] == lat2[i] && lat1[i] > 0);
  return x;
}

int build_lattice_diagonal(
    int *grid, std::vector<std::mt19937> &gens,
    std::vector<std::uniform_int_distribution<>> &indices, const int N,
    const int num_red, const int num_blue) {
  for (int i = 0; i < N * N * N; ++i)
    grid[i] = 0; // all sites vacant in the beginning
  int curr_red, curr_blue;
  curr_red = curr_blue = 0;
  const int tid = omp_get_thread_num();
  while (curr_red < num_red) {
    const int site = indices[tid](gens[tid]);
    auto [i, j, k] = revert(site, N);
    if ((i + j + k) % 2 == 0 && grid[site] == 0) {
      grid[site] = 1;
      curr_red++;
    }
  }
  while (curr_blue < num_blue) {
    const int site = indices[tid](gens[tid]);
    auto [i, j, k] = revert(site, N);
    if ((i + j + k) % 2 == 1 && grid[site] == 0) {
      grid[site] = 2;
      curr_blue++;
    }
  }

  int nred, nblue;
  nred = nblue = 0;
  for (int i = 0; i < N * N * N; ++i) {
    if (grid[i] == 1)
      nred++;
    else if (grid[i] == 2)
      nblue++;
  }

  assert(nred == num_red);
  assert(nblue == num_blue);
  return 0;
}

static inline void exchange(int *__restrict grid, const int &N, const int &site,
                            const int &to) {
  const auto tmp = grid[site];
  grid[site] = grid[to];
  grid[to] = tmp;
}

#define ONETWO(x) ((x & 1) || (x & 2))
static inline float local_energy(int *__restrict grid, const int &N,
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

static inline const float nn_energy(int *__restrict lattice, const int N,
                                    const int &i, const int &j, const int &k) {
  const auto site = k + N * (j + N * i);
  float res = local_energy(lattice, N, i, j, k);
  const auto nne = revert_neighbor_table[site];
  for (const auto &[m, l, n] : nne)
    res += local_energy(lattice, N, m, l, n);
  return res;
}

// single thread calls this function
void fully_nonlocal_sweep(int *__restrict lattice, const int L,
                          std::vector<std::mt19937> &gens,
                          std::vector<std::uniform_real_distribution<>> &unis,
                          std::vector<std::uniform_int_distribution<>> &indices,
                          const float beta) {
  // lattice points already to own lattice per thread
  // ie. lattices_per_rank[tid]
  const auto tid = omp_get_thread_num();
  for (int i = 0; i < L * L * L; ++i) {
    const int site = indices[tid](gens[tid]);
    const int mv = indices[tid](gens[tid]);
    if (lattice[site] == lattice[mv])
      continue;
    const auto [si, sj, sk] = revert_table[site];
    const auto [mi, mj, mk] = revert_table[mv];

    const float E1 =
        nn_energy(lattice, L, si, sj, sk) + nn_energy(lattice, L, mi, mj, mk);
    exchange(lattice, L, site, mv);
    const float E2 =
        nn_energy(lattice, L, si, sj, sk) + nn_energy(lattice, L, mi, mj, mk);
    const float dE = E2 - E1;
    if (dE <= 0 || unis[tid](gens[tid]) < std::exp(-beta * dE))
      continue;
    exchange(lattice, L, site, mv);
  }
}

void local_sweep(int *__restrict lattice, const int L,
                 std::vector<std::mt19937> &gens,
                 std::vector<std::uniform_real_distribution<>> &unis,
                 std::vector<std::uniform_int_distribution<>> &indices,
                 const float beta) {
  const auto tid = omp_get_thread_num();
  for (int i = 0; i < L * L * L; ++i) {
    // is this even better than taking modulo?
    const int site = indices[tid](gens[tid]);
    const auto nns = nn[site];
    // the below is equal to: const auto nb  = indices[tid](gens[tid]) % 6;
    // ie. maybe the compiler already sees this but to be safe we hand-code this
    // here
    const auto nb = indices[tid](gens[tid]) & 5;
    const auto mv = nns[nb]; // pick a neighbor
    if (lattice[site] == lattice[mv])
      continue;
    const auto [si, sj, sk] = revert_table[site];
    const auto [mi, mj, mk] = revert_table[mv];
    const float E1 =
        nn_energy(lattice, L, si, sj, sk) + nn_energy(lattice, L, mi, mj, mk);
    exchange(lattice, L, site, mv);
    const float E2 =
        nn_energy(lattice, L, si, sj, sk) + nn_energy(lattice, L, mi, mj, mk);
    const float dE = E2 - E1;
    if (dE <= 0 || unis[tid](gens[tid]) < std::exp(-beta * dE))
      continue;
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

int energy(int *__restrict lattice, const int &L) {
  int e = 0;
#pragma omp parallel for collapse(3) reduction(+ : e)
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k)
        e += local_energy(lattice, L, i, j, k);
  return e;
}

void collect_and_print(const int &rk, const int &sz, const int &num_threads,
                       int *__restrict all_configs,
                       int *__restrict rank_lattices, const int &padded_N,
                       const int &L) {
  MPI_Gather(rank_lattices, padded_N * num_threads, MPI_INT, all_configs,
             padded_N * num_threads, MPI_INT, 0, MPI_COMM_WORLD);
  if (rk == 0) {
    for (int r = 0; r < sz; ++r)
      for (int t = 0; t < num_threads; ++t)
        std::cout << energy(all_configs + (r * num_threads + t) * padded_N, L)
                  << " ";
    std::cout << "\n";
  }
}

void collect_and_dump(const int &rk, const int &sz, const int &num_threads,
                      int *__restrict all_configs,
                      int *__restrict rank_lattices, const int &padded_N,
                      const int &L) {
  MPI_Gather(rank_lattices, padded_N * num_threads, MPI_INT, all_configs,
             padded_N * num_threads, MPI_INT, 0, MPI_COMM_WORLD);
  if (rk == 0) {
    for (int r = 0; r < sz; ++r) {
      for (int t = 0; t < num_threads; ++t) {
        int *lattice = all_configs + (r * num_threads + t) * padded_N;
        for (int i = 0; i < L * L * L; ++i)
          std::cout << lattice[i] << " ";
        std::cout << "\n";
      }
    }
  }
}

void collect_and_dump_corr(const int &rk, const int &sz, const int &num_threads,
                           int *__restrict beg_configs,
                           int *__restrict all_configs,
                           int *__restrict rank_lattices, const int &padded_N,
                           const int &L) {
  MPI_Gather(rank_lattices, padded_N * num_threads, MPI_INT, all_configs,
             padded_N * num_threads, MPI_INT, 0, MPI_COMM_WORLD);
  if (rk == 0) {
    for (int r = 0; r < sz; ++r) {
      for (int t = 0; t < num_threads; ++t) {
        int *beg_lattice = beg_configs + (r * num_threads + t) * padded_N;
        int *lattice = all_configs + (r * num_threads + t) * padded_N;
        std::cout << logand(beg_lattice, lattice, L) << " ";
      }
    }
    std::cout << "\n";
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "run as ./bin L\n";
    return 1;
  }

  int rk, sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  const auto arg_n = argv[1];
  const auto L = atoi(arg_n);

  const float rho = .75;

  // technically, for the correlation function to check out, rho1 = .6 and rho2
  // = .4 but here we define them like this for convenience (else factor C_0 is
  // the wrong expected value of repeated Bernoulli trials)
  const float rho1 = .45; // .6 * rho
  const float rho2 = .3;  // .4 * rho
  const int numparts = static_cast<int>(rho * L * L * L);

  const int r = static_cast<int>(rho1 * L * L * L);
  const int b = numparts - r;

  assert(rho1 + rho2 == rho);

  // fix num_threads to 4 to adhere to constraints on Daint
  omp_set_num_threads(4);
  // allocate global LUTs to enhance computation
  ::nn = get_nn_list(L);
  ::revert_table = generate_revert_table(L);
  ::revert_neighbor_table = generate_neighbor_revert_table(L);

#pragma omp parallel
  assert(omp_get_num_threads() == 4);

  const int num_threads = 4;

  int *rank_lattices = nullptr;

  const unsigned int align = 128;
  const auto padded_N = (L * L * L + (align - 1)) & ~(align - 1);

  if (posix_memalign((void **)&rank_lattices, align,
                     num_threads * padded_N * sizeof(int)) != 0)
    return 1;

  std::vector<std::mt19937> thread_generators(num_threads);
  std::vector<std::uniform_real_distribution<>> unis(num_threads);
  std::vector<std::uniform_int_distribution<>> indices(num_threads);

#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
#pragma omp critical
    { thread_generators[tid].seed(tid); }
  }

  const int num_configs = num_threads * sz;
  float *betas = (float *)malloc(sizeof(float) * num_configs);
  const float beta_min = 1.;
  const float beta_max = 3.;
  const float d = (beta_max - beta_min) / num_configs;
  for (int i = 0; i < num_configs; ++i)
    betas[i] = beta_min + i * d;

#pragma omp parallel
  {
    // every thread allocates its own RNG
    const auto tid = omp_get_thread_num();
    const auto max_idx = L * L * L - 1;
#pragma omp critical
    {
      unis[tid] = std::uniform_real_distribution<>(0., 1.);
      indices[tid] = std::uniform_int_distribution<>(0, max_idx);
    }
  }

#pragma omp parallel
  {
    const int t = omp_get_thread_num();
#pragma omp critical
    build_lattice_diagonal(rank_lattices + t * padded_N, thread_generators,
                           indices, L, r, b);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  int *all_configs = nullptr;
  int *beg_configs = nullptr;
  if (rk == 0) {
    all_configs = (int *)malloc(sizeof(int) * padded_N * sz * num_threads);
    beg_configs = (int *)malloc(sizeof(int) * padded_N * sz * num_threads);
  }

  // collect all initial configs
  collect_and_print(rk, sz, num_threads, all_configs, rank_lattices, padded_N,
                    L);
  // collect_and_dump(rk, sz, num_threads, all_configs, rank_lattices, padded_N,
  // L);

  if (rk == 0) {
#pragma omp master
    {
      for (int r = 0; r < sz; ++r) {
        for (int t = 0; t < num_threads; ++t) {
          const int offset = (r * num_threads + t) * padded_N;
          const int *lattice = (all_configs + offset);
          for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
              for (int k = 0; k < L; ++k) {
                assert(lattice[k + L * (j + i * L)] < 3 &&
                       lattice[k + L * (j + i * L)] > -1);
              }
            }
          }
        }
      }
    }
  }

  int nsweeps = 1 << 10;

  int printer = 1;

#pragma omp parallel
  for (int i = 1; i <= nsweeps + 1; ++i) {
    {
      const int tid = omp_get_thread_num();
      const float my_beta = betas[rk * num_threads + tid];
      int *my_lattice = rank_lattices + tid * padded_N;
      fully_nonlocal_sweep(my_lattice, L, thread_generators, unis, indices,
                           my_beta);
    }

// unfortunately threads have to sync for every sweep because else they run off
// and we can't get them back causing massive errors
#pragma omp barrier

#pragma omp master
    if (i % printer == 0) {
      collect_and_print(rk, sz, num_threads, all_configs, rank_lattices,
                        padded_N, L);
      // collect_and_dump(rk, sz, num_threads, all_configs, rank_lattices,
      // padded_N,  L);
      printer *= 2;
      // have all ranks wait before continuing sweeps
      // minimal overhead to be safe
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  if (rk == 0)
#pragma omp master
    std::cout << "\n";

  printer = 1;
  const int local_nsweeps = 1000;

  // all <equilibrated> configs for us to check decorrelation periodically
  MPI_Gather(rank_lattices, padded_N * num_threads, MPI_INT, all_configs,
             padded_N * num_threads, MPI_INT, 0, MPI_COMM_WORLD);
  if (rk == 0)
#pragma omp master
    memcpy(beg_configs, all_configs, sizeof(int) * padded_N * sz * num_threads);

  collect_and_dump_corr(rk, sz, num_threads, beg_configs, all_configs,
                        rank_lattices, padded_N, L);

#pragma omp parallel
  for (int i = 1; i <= local_nsweeps + 1; ++i) {
    {
      const int tid = omp_get_thread_num();
      const float my_beta = betas[rk * num_threads + tid];
      int *my_lattice = rank_lattices + tid * padded_N;
      local_sweep(my_lattice, L, thread_generators, unis, indices, my_beta);
    }

// unfortunately threads have to sync for every sweep because else they run off
// and we can't get them back causing massive errors
#pragma omp barrier

#pragma omp master
    if (i % printer == 0) {
      collect_and_dump_corr(rk, sz, num_threads, beg_configs, all_configs,
                            rank_lattices, padded_N, L);
      // collect_and_print(rk, sz, num_threads, all_configs, rank_lattices,
      //                   padded_N, L);
      // collect_and_dump(rk, sz, num_threads, all_configs, rank_lattices,
      // padded_N,  L);
      printer *= 2;
      // have all ranks wait before continuing sweeps
      // minimal overhead to be safe
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  free(rank_lattices);
  free(betas);
  if (rk == 0)
    free(all_configs);
  MPI_Finalize();
  return 0;
}
