// mpicxx -pg -pedantic -ffast-math -march=native -O3 -Wall -fopenmp -Wunknown-pragmas  -lm -lstdc++ -std=c++17 mpi_mc.cpp -o to_mpi

// clang++-17 -pedantic -ffast-math -march=native -O3 -Wall -fopenmp
// -Wunknown-pragmas -fsanitize=address -lm -lstdc++ -std=c++20 simple_mc.cpp -o
// to_simple; time ./to_simple
//
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
void print_slice(int *lattice, int N, int i) {
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k)
      std::cout << lattice[k + N * (j + i * N)] << " ";
    std::cout << "\n";
  }
}

std::random_device rd_sample;
std::mt19937 gen_sample(__rdtsc());
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

  float connection = (float)(grid[site] == 1 ? 3 : 5);
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
#pragma omp parallel for collapse(1) firstprivate(gen, dis, uni) schedule(static)
    for (int i = ii; i < N - 3 + 1; i += 3) {
      int m = (i > 2 ? 2 : 0); 
      //int m=0;
      for (int j = jj; j < N - 3 + 1 + m; ++j)
      {
             
        int n = (i > 2? 2 : 0);
        //int n=0;
        for (int k = kk; k < N - 3 + 1 + n; ++k) {
          const auto site = k + N * (j + i * N);
          
          if (lattice[site] == 0) continue;
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
                                  
          if (lattice[mv] == lattice[site]) continue; // same particle type will not change energy 

          float E1 = local_energy(lattice, N, i, j, k);
          auto [mi, mj, mk] = revert(mv, N);
          exchange(lattice, N, site, mv);
          float E2 = local_energy(lattice, N, i, j, k) +
                     local_energy(lattice, N, mi, mj, mk);
          
          float dE = E2 - E1;
          if (uni(gen) >= std::exp(-beta * dE))
            continue; // success, accept new state
          exchange(lattice, N, mv, site);
        }
      }
    }
}

float energy(int *__restrict lattice, const int &N) {
  float e = 0.;
#pragma omp parallel for collapse(3) reduction(+ : e)
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        e += local_energy(lattice, N, i, j, k);
  return e;
}

int main() {
  int N = 30;
  int r = 12000;
  int b = 8000;

  float beta = 3.;

  int *lattice = nullptr;
  const unsigned int align = 64;
  auto padded_N = (N * N * N + (align - 1)) & ~(align - 1);
  if (posix_memalign((void **)&lattice, align, padded_N * sizeof(int)) != 0)
    return 1;
  build_lattice(lattice, N, r, b);

  std::cout << "scale in 1e6 epochs\n";
  int nsteps = 10000;

  float curr = 0.;
  omp_set_num_threads(10); // 10 performs the best so far
  for (int i = 0; i < nsteps; ++i) {
    auto t0 = __rdtsc(); // rather use MPI_Wtime()
    sweep(lattice, N, beta);
    auto tf = __rdtsc();
    curr += tf - t0;
  }
  std::cout << "OMP with " << 10 << " cores takes ~" << curr / nsteps / 1e6 << "*1e6 cycles / sweep / proc\n";


  free(lattice);
}
