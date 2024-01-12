#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>

#include <random>
#include <stdexcept>
#include <stdlib.h>

#include <immintrin.h>
#include <unordered_map>

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
        const auto idx = k + N * (j + i * N);
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
    const int rand = dis(gen_sample);
    if (grid[rand] == 0)
      return rand;
  }
}

int sample(const int *__restrict a, const int &N, const int &count,
           const int &part_type) {
  std::uniform_int_distribution<> dis(0, N - 1);
  int res = -1;
  while (res == -1) {
    const auto idx = dis(gen_sample);
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

std::array<int, 6> get_neighbors(const int &i, const int &j, const int &k,
                                 const int &N) {
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

static inline float local_energy(int *__restrict grid, const int &N,
                                 const int &i, const int &j, const int &k) {
  const auto site = k + N * (j + N * i);
  if (grid[site] == 0)
    return 0.;

  const float connection = grid[site] == 1 ? 3. : 5.;
  float current = 0.;

  const auto [l, r, u, d, f, b] = nn[site];

  // load all nearest neighbors of site into 1 _mm256
  __m256 v = _mm256_set_ps(grid[u] & 1, grid[d] & 1, grid[l] & 1, grid[r] & 1,
                           grid[f] & 1, grid[b] & 1, 0., 0.);
  // lowest 4 nearest neighbors into _m128
  __m128 v_hi = _mm256_extractf128_ps(v, 1);
  // highest 4 nearest neighbors into _m128
  __m128 v_lo = _mm256_extractf128_ps(v, 0);
  // add lowest and highest
  v_lo = _mm_add_ps(v_lo, v_hi);
  // shuffle around
  v_lo = _mm_add_ss(v_lo, _mm_shuffle_ps(v_lo, v_lo, 1));
  // unpack sum of all nn into `current` float
  current = _mm_cvtss_f32(v_lo);

  current = connection - current;
  return current * current;
}

void sweep(int *__restrict lattice, const int N, const float beta = 1.) {
#pragma omp parallel for collapse(2) schedule(auto) shared(lattice)            \
    firstprivate(N)
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
              const auto nn = ::nn[site];
              const auto mv = nn[dis(gen)]; // get one of the site's neighbors

              if (lattice[mv] == lattice[site])
                continue; // same particle type will not change energy
                          // optimization that should not induce bias

              const float E1 = local_energy(lattice, N, i, j, k);
              // auto [mi, mj, mk] = revert(mv, N);
              exchange(lattice, N, site, mv);
              const float E2 = local_energy(lattice, N, i, j, k);

              const float dE = std::abs(E2 - E1);
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
  if (argc != 3) {
    std::cout << "run as ./bin beta num_threads\n";
    return 1;
  }
  const auto arg_beta = argv[1];
  const auto beta = atof(arg_beta);
  const auto arg_nsw = argv[2];
  const auto nsw = atoi(arg_nsw);

  omp_set_num_threads(nsw);

  int N = 30;
  int r = 12000;
  int b = 8000;
  //
  //  ~ values available in PLR
  //const int N = 20;
  //const int r = 3556;
  //const int b = 2370;

  // int N = 15;
  // int r = 1518;
  // int b = 1012;

  nn = get_nn_list(N);

  int *lattice = nullptr;
  int *lattice_back = nullptr;
  int *cpy_lattice = nullptr;
  const unsigned int align = 64;
  const auto padded_N = (N * N * N + (align - 1)) & ~(align - 1);

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

  const int nsweeps = (1 << 17) + 1;
  std::vector<std::tuple<int, int, int>> data;

  data.push_back({0, logand(lattice_back, lattice, N), 0});

  int curr = 1;
 
  
  for (int i = 1; i <= nsweeps; ++i) {
    sweep(lattice, N, beta);
    if (i == curr) { 
      data.push_back({i, logand(lattice_back, lattice, N), 0});
      curr *= 2;
    }
    }

  free(lattice);
  free(lattice_back);
  return 0;
}
