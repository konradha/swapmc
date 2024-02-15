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

int logand(const short *__restrict lattice1, const short *__restrict lattice2,
           const int &N) {
  int log = 0;
#pragma omp parallel for collapse(3) reduction(+ : log)
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k) {
        const auto idx = k + N * (j + i * N);
        log += (lattice1[idx] == lattice2[idx]) && lattice1[idx];
      }
  return log;
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

// we allow: swaps and moves to empty slots
static inline void step(short *__restrict lattice, const int N, const int &i,
                        const int &j, const int &k,
                        std::vector<std::mt19937> &gens,
                        std::vector<std::uniform_int_distribution<>> &nn_dis,
                        std::vector<std::uniform_real_distribution<>> &unis,
                        const float beta) {
  const auto t = omp_get_thread_num();
  const auto nthreads = omp_get_num_threads();
  const auto site = k + N * (j + i * N);
  if (lattice[site] == 0)
    return;

  const auto nn = ::nn[site];
  const auto to_mv = nn_dis[t](gens[t]);

  const auto mv = nn[to_mv];
  auto [mi, mj, mk] = revert(mv, N);

  const volatile float E1 =
      nn_energy(lattice, N, i, j, k) + nn_energy(lattice, N, mi, mj, mk);
  exchange(lattice, N, site, mv);
  const volatile float E2 =
      nn_energy(lattice, N, i, j, k) + nn_energy(lattice, N, mi, mj, mk);
  const auto dE = (E2 - E1);
  const auto crit = (std::exp(-beta * dE) < 1. ? std::exp(-beta * dE) : 1.);

  if (unis[t](gens[nthreads + t]) < crit) {
    return;
  } else {
    exchange(lattice, N, site, mv);
  }
}

void slice_sweep_randomized(
    short *__restrict lattice, const int N, const int &i,
    std::vector<std::mt19937> &gens,
    std::vector<std::uniform_int_distribution<>> &nn_dis,
    std::vector<std::uniform_real_distribution<>> &unis,
    std::vector<std::vector<std::tuple<int, int>>> &slice_coords,
    const float beta) {
  {
    const auto tid = omp_get_thread_num();
    std::shuffle(slice_coords[tid].begin(), slice_coords[tid].end(), gens[tid]);
    for (const auto &[j, k] : slice_coords[tid]) {
      step(lattice, N, i, j, k, gens, nn_dis, unis, beta);
    }
  }
}

void sweep(short *__restrict lattice, const int N,
           std::vector<std::mt19937> &gens,
           std::vector<std::uniform_int_distribution<>> &nn_dis,
           std::vector<std::uniform_real_distribution<>> &unis,
           const float beta,
           std::vector<std::vector<std::tuple<int, int>>> &slice_coords) {
  {
    const int stride = 4;
    const int mod = N % stride;
    assert(mod == 0);

    std::vector<int> idx = {0, 1, 2, 3};
    int ii = 0;
#pragma omp master
    std::shuffle(idx.begin(), idx.end(), gens[0]);

#pragma omp parallel shared(idx)
    for (int s = 0; s < stride; ++s) {
#pragma omp barrier // everyone needs access to same s
      ii = idx[s];
#pragma omp for collapse(1) schedule(auto)
      for (int i = ii; i < N; i += stride) {
        slice_sweep_randomized(lattice, N, i, gens, nn_dis, unis, slice_coords,
                               beta);
      }
    }
  }
}

float energy(short *__restrict lattice, const int &L) {
  float e = 0.;
#pragma omp parallel for collapse(3) reduction(+ : e)
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k)
        e += local_energy(lattice, L, i, j, k);
  return e;
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

void dump_copies(short *copies, const int &L) {
  int rk, sz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  if (rk != 0)
    return;

  for (int i = 0; i < sz; ++i) {
    for (int d = 0; d < L * L * L; ++d)
      std::cout << copies[i * L * L * L + d] << " ";
    std::cout << "\n";
  }
}

void try_exchange(short *lattice, const int &L, short *copies,
                  const int &flag) {
  int rk, sz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);
  const int N = L * L * L;

  float rank_energy, partner_energy;
  rank_energy = energy(lattice, L);

  float min_beta = .1;
  float max_beta = 4.;
  float d_beta = (max_beta - min_beta) / (sz - 1);

  int exchange_flag = -1;
  if (flag) {
    int partner = -1;
    if (rk % 2 == 0) {
      partner = rk + 1;
      const float bi = min_beta + rk * d_beta;
      const float bj = min_beta + partner * d_beta;

      MPI_Recv(&partner_energy, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      const float dE = std::exp((-bj * rank_energy - bi * partner_energy) /
                                (-bi * rank_energy - bj * partner_energy));

      exchange_flag = (int)(dE > std::rand() / (float)(RAND_MAX));

      MPI_Send(&exchange_flag, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);

      if (exchange_flag) {
        MPI_Sendrecv_replace(lattice, N, MPI_SHORT, partner, 0, partner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      // exchange debugging statement
      // std::cout << rk << " <-> " << partner << "\n";
      // if( exchange_flag && rk == 0) std::cout << "rank 0 exchanged\n";
    } else {
      partner = rk - 1;
      MPI_Send(&rank_energy, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);

      MPI_Recv(&exchange_flag, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      if (exchange_flag) {
        MPI_Sendrecv_replace(lattice, N, MPI_SHORT, partner, 0, partner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }

  else {
    int partner = -1;
    if (rk == 0 || rk == sz - 1)
      return;

    if (rk % 2 == 1) {
      partner = rk + 1;

      const float bi = min_beta + rk * d_beta;
      const float bj = min_beta + partner * d_beta;
      MPI_Recv(&partner_energy, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      const float dE = std::exp((-bj * rank_energy - bi * partner_energy) /
                                (-bi * rank_energy - bj * partner_energy));

      exchange_flag = (int)(dE > std::rand() / (float)(RAND_MAX));

      MPI_Send(&exchange_flag, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);

      if (exchange_flag) {
        MPI_Sendrecv_replace(lattice, N, MPI_SHORT, partner, 0, partner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      // exchange debugging statement
      // std::cout << rk << " <-> " << partner << "\n";
    } else {
      partner = rk - 1;
      MPI_Send(&rank_energy, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);

      MPI_Recv(&exchange_flag, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      if (exchange_flag) {
        MPI_Sendrecv_replace(lattice, N, MPI_SHORT, partner, 0, partner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }
}

void seed(int k) { srand(time(NULL) + k); }

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "run as ./bin num_threads L\n";
    return 1;
  }

  int rk, sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  // seed all ranks differently
  seed(rk);

  const auto arg_nsw = argv[1];
  const auto nsw = atoi(arg_nsw);
  const auto arg_n = argv[2];
  const auto L = atoi(arg_n);

  const int N = L;
  const float rho = .75;
  const float rho1 = .4 * rho;
  const float rho2 = rho - rho1;
  const int numparts = static_cast<int>(rho * L * L * L);

  const int r = static_cast<int>(rho1 * L * L * L);
  const int b = numparts - r;

  assert(rho1 + rho2 == rho);

  omp_set_num_threads(nsw);
  ::nn = get_nn_list(L);

  short *lattice = nullptr;
  const unsigned int align = 64;
  const auto padded_N = (L * L * L + (align - 1)) & ~(align - 1);

  if (posix_memalign((void **)&lattice, align, padded_N * sizeof(short)) != 0)
    return 1;

  // create and distribute lattice to all ranks
  if (rk == 0)
    build_lattice(lattice, L, r, b);

  MPI_Bcast(lattice, L * L * L, MPI_SHORT, 0, MPI_COMM_WORLD);

  const int nthreads = get_num_threads();
  std::vector<std::mt19937> thread_generators(2 * nthreads);

#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
#pragma omp critical
    {
      thread_generators[tid].seed(std::random_device()() + tid);
      thread_generators[omp_get_num_threads() + tid].seed(
          std::random_device()() + 100 * tid);
    }
  }

  std::vector<std::uniform_int_distribution<>> nn_dis(nthreads);
  std::vector<std::uniform_real_distribution<>> unis(nthreads);

#pragma omp parallel
  {
    // every thread allocates its own RNG
    const auto tid = omp_get_thread_num();
    if (tid == 0)
      for (int t = 0; t < omp_get_num_threads(); ++t) {
        nn_dis[t] = std::uniform_int_distribution<>(0, 5);
        unis[t] = std::uniform_real_distribution<>(0., 1.);
      }
  }

  std::vector<std::vector<std::tuple<int, int>>> slice_coords(nthreads);
#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k)
#pragma omp critical
        slice_coords[tid].emplace_back(j, k);

    std::shuffle(slice_coords[tid].begin(), slice_coords[tid].end(),
                 thread_generators[tid]);
    std::sort(slice_coords[tid].begin(), slice_coords[tid].end());
  }

  // distribute configs to all ranks
  // start parallel tempering loop:
  // 1. gather and save
  // 2. decide for strategy [flag]
  // 3. - if flag: (0,1) (2,3) ...
  //    - else:    (1,2) (3,4) ... exclude {0, sz-1}
  // 4. first one gets other's energy
  // 5. then signal upon metropolis criterion
  // 6. sendrecv_replace

  short *copies = nullptr;
  if (rk == 0) {
    copies = (short *)malloc(sizeof(short) * L * L * L * sz);
  }

  const int nsweeps = (1 << 20) + 1;

  const auto beta_min = .1;
  const auto beta_max = 4.;
  const auto db = (beta_max - beta_min) / (sz - 1);
  const float rank_beta = beta_min + rk * db;

  int curr = 1;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> uni(0., 1.);
  assert(L % 4 == 0 && "L must be a factor of 4!");

  for (int i = 1; i <= nsweeps; ++i) {
    sweep(lattice, N, thread_generators, nn_dis, unis, rank_beta, slice_coords);
#pragma omp master
    {
      auto red = sample(lattice, N * N * N, r, 1);
      auto blue = sample(lattice, N * N * N, b, 2);

      auto [ii, j, k] = revert(red, N);
      auto [x, y, z] = revert(blue, N);

      float E1 =
          nn_energy(lattice, N, ii, j, k) + nn_energy(lattice, N, x, y, z);
      exchange(lattice, N, red, blue);
      float E2 =
          nn_energy(lattice, N, ii, j, k) + nn_energy(lattice, N, x, y, z);
      auto dE = E2 - E1;
      if (uni(gen) < std::exp(-rank_beta * dE)) {
        exchange(lattice, N, red, blue);
      }
    }

    if (i == curr && sz > 1) {
      // get all configurations into `copies`
      MPI_Gather(lattice, L * L * L, MPI_SHORT, copies, L * L * L, MPI_SHORT, 0,
                 MPI_COMM_WORLD);
      if (rk == 0) dump_copies(copies, L);
      // if (rk == 0) std::cout << "sweep " << i << "\n";

      //if (rk == 0) {
      //  std::cout << i << ",";
      //  for (int i = 0; i < sz; ++i) {
      //    if (i < sz - 1)
      //      std::cout << energy(copies + i * L * L * L, L) << ",";
      //    else
      //      std::cout << energy(copies + i * L * L * L, L) << "\n";
      //  }
      //}
      // decide on a common strategy to attempt replica exchanges across
      // neighboring temperatures
      int flag = std::rand() & 1;
      MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // try to exchange configurations between neighboring temperatures
      try_exchange(lattice, L, copies, flag);
      curr *= 2;
    }
  }

  free(lattice);
  MPI_Finalize();
  return 0;
}
