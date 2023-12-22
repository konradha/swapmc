// mpicxx -ffast-math -march=native -O3 -Wall -Wunknown-pragmas -fopenmp  -lm
// -lstdc++ -std=c++17  mpi_mc.cpp -o to_mpi

// mpicxx -pg -pedantic -ffast-math -march=native -O3 -Wall -fopenmp
// -Wunknown-pragmas  -lm -lstdc++ -std=c++17 mpi_mc.cpp -o to_mpi

// to benchmark
// export OMP_PLACES=threads; clang++-17 -pg -pedantic -ffast-math -march=native
// -O3 -Wall -fopenmp -Wunknown-pragmas -fsanitize=address -lm -lstdc++
// -std=c++20 simple_mc.cpp -o to_simple; time ./to_simple

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>

#include <random>
#include <stdexcept>
#include <stdlib.h>

// can do along axis
void print_slice(const int *__restrict lattice, const int N, const int i) {
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k)
      std::cout << lattice[k + N * (j + i * N)] << " ";
    std::cout << "\n";
  }
}

void dump_slice(const int *__restrict lattice, const int N, const int i) {
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k)
      std::cout << lattice[k + N * (j + i * N)] << ",";
  }
}

void dump_lattice(const int *__restrict lattice, int N) {
  for (int i = 0; i < N; ++i)
    dump_slice(lattice, N, i);
  std::cout << "\n";
}

std::random_device rd_sample;
std::mt19937 gen_sample(33); // TODO adapt to MPI
int sample_vacant(int *grid, int sz) {
  std::uniform_int_distribution<> dis(0, sz - 1);
  // our algorithm assumes that the number of vacant spots is always larger than
  // num_reds + num_blue
  for (;;) {
    int rand = dis(gen_sample);
    if (grid[rand] == 0)
      return rand;
  }
}

int build_lattice(int *grid, const int N, const int num_red,
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

static inline void exchange(int *__restrict grid, const int &N, const int &site,
                            const int &to) {
  auto tmp = grid[site];
  grid[site] = grid[to];
  grid[to] = tmp;
}

static inline std::tuple<int, int, int> revert(int s, int N = 30) {
  auto k = s % N;
  auto j = ((s - k) / N) % N;
  auto i = (s - k - j * N) / (N * N);
  return {i, j, k};
}

static inline float local_energy(int *__restrict grid, const int &N,
                                 const int &i, const int &j, const int &k) {
  auto site = k + N * (j + N * i);
  if (grid[site] == 0)
    return 0.;

  float connection = grid[site] == 1 ? 3. : 5.;
  float current = 0.;
  int l, r, u, d, f, b; // left, right, up, down, front, back
  if (i == 0) {
    u = k + N * (j + (N - 1) * N);
  }
  if (i == N - 1) {
    d = k + N * j;
  }
  if (j == 0) {
    l = k + N * ((N - 1) + i * N);
  }
  if (j == N - 1) {
    r = k + N * (i * N);
  }
  if (k == 0) {
    f = N - 1 + N * (j + i * N);
  }
  if (k == N - 1) {
    b = N * (j + i * N);
  }
  if (i > 0) {
    u = k + N * (j + (i - 1) * N);
  }
  if (i < N - 1) {
    d = k + N * (j + (i + 1) * N);
  }
  if (j > 0) {
    l = k + N * (j - 1 + i * N);
  }
  if (j < N - 1) {
    r = k + N * (j + 1 + i * N);
  }
  if (k > 0) {
    f = k - 1 + N * (j + i * N);
  }
  if (k < N - 1) {
    b = k + 1 + N * (j + i * N);
  }

#pragma unroll(6)
  for (const auto &nn : {u, d, l, r, f, b}) {
    if (!grid[nn]) // no contributions from empty sites
      continue;
    current += 1.;
  }
  current = connection - current;
  return current * current;
}

void sweep(int *__restrict lattice, const int N, const float beta = 1.) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 5);
  std::uniform_real_distribution<> uni(0., 1.);

  int ii, jj, kk;
  ii = jj = kk = 0;
  for (ii = 0; ii < 3; ++ii) // distance at which there is
                             // no race between threads
                             // -> kind of a checkerboard flip
#pragma omp parallel for collapse(1) firstprivate(gen, dis, uni)               \
    schedule(static)
    for (int i = ii; i < N - 3 + 1; i += 3) {
      int m = (i > 2 ? 2 : 0);
      // int m=0;
      for (int j = jj; j < N - 3 + 1 + m; ++j) {

        int n = (i > 2 ? 2 : 0);
        // int n=0;
        for (int k = kk; k < N - 3 + 1 + n; ++k) {
          const auto site = k + N * (j + i * N);

          if (lattice[site] == 0)
            continue;
          int l, r, u, d, f, b; // left, right, up, down, front, back
          l = r = u = d = f = b = 0;
          if (i == 0) {
            u = k + N * (j + (N - 1) * N);
          }
          if (i == N - 1) {
            d = k + N * j;
          }
          if (j == 0) {
            l = k + N * ((N - 1) + i * N);
          }
          if (j == N - 1) {
            r = k + N * (i * N);
          }
          if (k == 0) {
            f = N - 1 + N * (j + i * N);
          }
          if (k == N - 1) {
            b = N * (j + i * N);
          }

          if (i > 0) {
            u = k + N * (j + (i - 1) * N);
          }
          if (i < N - 1) {
            d = k + N * (j + (i + 1) * N);
          }
          if (j > 0) {
            l = k + N * (j - 1 + i * N);
          }
          if (j < N - 1) {
            r = k + N * (j + 1 + i * N);
          }
          if (k > 0) {
            f = k - 1 + N * (j + i * N);
          }
          if (k < N - 1) {
            b = k + 1 + N * (j + i * N);
          }

          // assert(l >= 0 && l < N*N*N && "it's l");
          // assert(r >= 0 && r < N*N*N && "it's r");
          // assert(u >= 0 && u < N*N*N && "it's u");
          // assert(d >= 0 && d < N*N*N && "it's d");
          // assert(f >= 0 && f < N*N*N && "it's f");
          // assert(b >= 0 && b < N*N*N && "it's b");

          int nn[6] = {u, d, l, r, f, b};
          auto mv = nn[dis(gen)]; // get one of the site's neighbors

          if (lattice[mv] == lattice[site])
            continue; // same particle type will not change energy

          float E1 = local_energy(lattice, N, i, j, k);
          auto [mi, mj, mk] = revert(mv, N);
          exchange(lattice, N, site, mv);
          float E2 = local_energy(lattice, N, i, j, k) +
                     local_energy(lattice, N, mi, mj, mk);

          float dE = E2 - E1;
          if (uni(gen) <= std::exp(-beta * dE))
            continue; // success, accept new state
          exchange(lattice, N, mv, site);
        }
      }
    }
}

float energy(int *__restrict lattice, const int &N) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  float e = 0.;
#pragma omp parallel for collapse(3) reduction(+ : e)
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        e += local_energy(lattice, N, i, j, k);
  return e;
}

int *distribute_lattice(const int &N, const int &r, const int &b) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank != 0)
    return nullptr;

  int *lattice = nullptr;
  const unsigned int align = 64;
  auto padded_N = (N * N * N + (align - 1)) & ~(align - 1);
  if (posix_memalign((void **)&lattice, align, padded_N * sizeof(int)) != 0)
    return nullptr;
  build_lattice(lattice, N, r, b);
  int res = MPI_Bcast(lattice, N * N * N, MPI_INT, 0, MPI_COMM_WORLD);
  return lattice;
}

int *receive_start(const int &N, const int &r, const int &b) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    return nullptr;
  int *lattice = nullptr;
  const unsigned int align = 64;
  auto padded_N = (N * N * N + (align - 1)) & ~(align - 1);
  if (posix_memalign((void **)&lattice, align, padded_N * sizeof(int)) != 0)
    return nullptr;
  int res = MPI_Bcast(lattice, N * N * N, MPI_INT, 0, MPI_COMM_WORLD);
  return lattice;
}

float sweep_loop(int *__restrict lattice, const int &N, const float &beta,
                 const int &nsteps, double &t0, double &tf) {
  t0 += MPI_Wtime();
  for (int i = 0; i < nsteps; ++i)
    sweep(lattice, N, beta);   // parallel loop per rank
  auto e = energy(lattice, N); // parallel loop per rank
  tf += MPI_Wtime();
  return e;
}

int determine_and_distribute(const float &e, float *energies, const int &rk,
                             const int &sz, int *__restrict lattice,
                             const int &N, double &t0, double &tf) {
  t0 += MPI_Wtime();
  MPI_Gather(&e, 1, MPI_INT, energies, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int broadcaster = sz + 10;
  if (rk == 0) {
    float m = 1000000000.;
    for (int r = 0; r < sz; ++r)
      if (energies[r] < m) {
        m = energies[r];
        broadcaster = r;
      }
    MPI_Bcast(&broadcaster, 1, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(&broadcaster, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(lattice, N * N * N, MPI_INT, broadcaster, MPI_COMM_WORLD);
  tf += MPI_Wtime();
  return 0;
}

int main(int argc, char **argv) {
  int rk, sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);
  auto arg_beta = argv[1];
  auto beta = atof(arg_beta);

  int N = 30;
  int r = 12000;
  int b = 8000;

  int *lattice;
  if (rk == 0) {
    lattice = distribute_lattice(N, r, b);
    dump_lattice(lattice, N);
  } else
    lattice = receive_start(N, r, b);
  MPI_Barrier(MPI_COMM_WORLD);

  // omp_set_num_threads(4);
  int nsteps = 1;
  int ntries = 10000;
  double t0, tf;
  t0 = tf = 0.;
  double t0_comm, tf_comm;
  t0_comm = t0_comm = 0.;
  float *energies = (rk == 0 ? ((float *)calloc(sz, sizeof(float))) : nullptr);
  for (int i = 0; i < ntries; ++i) {
    auto e = sweep_loop(lattice, N, beta, nsteps, t0, tf);
    if (rk == 0)
      dump_lattice(lattice, N);
    determine_and_distribute(e, energies, rk, sz, lattice, N, t0_comm, tf_comm);
  }
  ////std::cout << rk << ": " << tf_comm - t0_comm  << " secs spent
  ///communicating / waiting\n"; /std::cout << rk << ": took " << tf - t0  << "
  ///secs for " << ntries <<  "*" << nsteps << " sweeps\n"; /std::cout << rk <<
  ///": took " << (tf - t0) / ntries  << " secs for " << nsteps << " sweeps (in
  ///average)\n";

  free(lattice);
  MPI_Finalize();
  return 0;
}
