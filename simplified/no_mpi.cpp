// clang++-17 -fsanitize=address -pedantic -fno-omit-frame-pointer -ggdb -ffast-math -march=native -O3 -Wall -Wunknown-pragmas -fopenmp  -lm -lstdc++ -std=c++17 no_mpi.cpp -o to_none

// benchmark'd with for i in $(seq 1 8); do GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./to_none 1. $i; done
#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>

#include <random>
#include <stdexcept>
#include <stdlib.h>


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

int logand(const int *__restrict lattice1, const int *__restrict lattice2,
           const int &N) {
  int log = 0;
#pragma omp parallel for collapse(3) reduction(+ : log)
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k) {
        auto idx = k + N * (j + i * N);
        log += (lattice1[idx] == lattice2[idx]) && lattice2[idx];
      }
  return log;
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

int sample(const int *__restrict a, const int &N, const int &count,
           const int &part_type) {
  std::uniform_int_distribution<> dis(0, N - 1);
  int res = -1;
  while (res == -1) {
    auto idx = dis(gen_sample);
    if (a[idx] == part_type)
      res = idx;
  }
  return res;
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

#pragma omp simd
  for (const auto &nn : {u, d, l, r, f, b}) {
    if (!grid[nn]) // no contributions from empty sites
      continue;
    current += 1.;
  }
  current = connection - current;
  return current * current;
}


void sweep(int *__restrict lattice, const int N, const float beta = 1.) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < N; ii += 3) {
    for (int jj = 0; jj < N; jj += 3) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(0, 5);
      std::uniform_real_distribution<> uni(0., 1.);
      for (int kk = 0; kk < N; kk += 3) {
        for (int i = ii; i < std::min(ii + 3, N); i++) {
          for (int j = std::max(0, jj - ii % 3); j < std::min(jj + 3, N); j++) {
            for (int k = std::max(0, kk - ii % 3); k < std::min(kk + 3, N);
                 k++) {
              const auto site = k + N * (j + i * N);

              // necessary for the local move dynamics
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
              int nn[6] = {u, d, l, r, f, b};
              auto mv = nn[dis(gen)]; // get one of the site's neighbors

              if (lattice[mv] == lattice[site])
                continue; // same particle type will not change energy
                          // optimization that should not induce bias

              float E1 = local_energy(lattice, N, i, j, k);
              // auto [mi, mj, mk] = revert(mv, N);
              exchange(lattice, N, site, mv);
              float E2 = local_energy(
                  lattice, N, i, j, k); // +
                                        // local_energy(lattice, N, mi, mj, mk);

              float dE = std::abs(E2 - E1);
              if (uni(gen) <= std::exp(-beta * dE)) {
                continue; // success, accept new state
              }
              exchange(lattice, N, site, mv); // change back what was swapped
            }
          }
        }
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

int main(int argc, char **argv) {
  if (argc != 3)
  {
    std::cout << "run as ./bin beta num_threads\n";
    return 1;
  }


  auto arg_beta = argv[1];
  auto beta = atof(arg_beta);

  auto arg_nsw = argv[2];
  auto nsw = atoi(arg_nsw);


  omp_set_num_threads(nsw);

  // int N = 30;
  // int r = 12000;
  // int b = 8000;
  //
  // ~ values available in PLR
  // int N = 20;
  // int r = 3556;
  // int b = 2370;

  int N = 15;
  int r = 1518;
  int b = 1012;


  int *lattice = nullptr;
  int *lattice_back = nullptr;
  const unsigned int align = 64;
  auto padded_N = (N * N * N + (align - 1)) & ~(align - 1);

  if (posix_memalign((void **)&lattice, align, padded_N * sizeof(int)) != 0)
    return 1;
  build_lattice(lattice, N, r, b);

  if (posix_memalign((void **)&lattice_back, align, padded_N * sizeof(int)) !=
      0)
    return 1;

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        lattice_back[k + N * (j + i * N)] = lattice[k + N * (j + i * N)];

  int nsweeps = (1 << 15) + 1;


  int curr = 1;
  std::uniform_real_distribution<> uni(0., 1.);
  auto t1 = __rdtsc();
  for (int i = 1; i <= nsweeps; ++i) {

    // if (uni(rd_sample) <= .33)
    {
      auto red = sample(lattice, N * N * N, r, 1);
      auto blue = sample(lattice, N * N * N, b, 2);
      auto [i, j, k] = revert(red, N);
      auto [x, y, z] = revert(blue, N);

      float E1 =
          local_energy(lattice, N, i, j, k) + local_energy(lattice, N, x, y, z);
      exchange(lattice, N, red, blue);
      float E2 =
          local_energy(lattice, N, i, j, k) + local_energy(lattice, N, x, y, z);
      auto dE = std::abs(E1 - E2);
      if (uni(rd_sample) >= std::exp(-beta * dE)) {
        exchange(lattice, N, red, blue);
      }
    }

    sweep(lattice, N, beta);

    if (i == curr) {
      logand(lattice_back, lattice, N);
      //std::cout << i << "," << logand(lattice_back, lattice, N) << "\n";
      curr *= 2;
    }
  }
  auto t2 = __rdtsc();
  std::cout << nsw << ","<< (float)(t2 - t1)/(1e6) << "\n";
  free(lattice);
  free(lattice_back);

  return 0;
}
