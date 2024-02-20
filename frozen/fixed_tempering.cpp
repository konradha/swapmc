/*
 * L=12, 2^25 sweeps, 8 temperatures:
 * real	378m7,275s
 * user	5663m39,750s
 * sys	6m0,481s
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
    //const auto tid = omp_get_thread_num();
    // std::shuffle(slice_coords[tid].begin(), slice_coords[tid].end(),
    // gens[tid]);

    // for (const auto &[j, k] : slice_coords[tid]) {
    //   step(lattice, N, i, j, k, gens, nn_dis, unis, beta);
    // }
    for (int s = 0; s < N * N; ++s) {
      const auto j = std::rand() % N;
      const auto k = std::rand() % N;
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

void sweep_fully_random(short * lattice, const int & L, const float & beta,
                        std::vector<std::mt19937> &gens,
                        std::vector<std::uniform_int_distribution<>> &nn_dis,
                        std::vector<std::uniform_real_distribution<>> &unis)
{
    for(int t=0;t<L*L*L;++t)
    {
        const int pos = std::rand() % (L * L * L);
        const auto [i, j, k] = revert(pos, L);
        step(lattice, L, i, j, k, gens, nn_dis, unis, beta);
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


void try_exchange_all(short *lattice, const int &L, short *copies,
                  float *all_energies, float *betas, int *betas_idx) {
  int rk, sz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  float rank_energy = energy(lattice, L);
  MPI_Gather(&rank_energy, 1, MPI_FLOAT, all_energies, 1, MPI_FLOAT, 0,
             MPI_COMM_WORLD);

  if (rk == 0)
  {
      const int f = std::rand() & 1;
      if (f)
      {
          // (0,1), (2,3) ...

          for(int i=0;i<sz;i+=2)
          {
            int changer, partner; changer = partner = -1;
            for (int j=0;j<sz;++j)
            {
                if (betas_idx[j] == i) changer = j;
                if (betas_idx[j] == i + 1) partner = j;
                if (partner != -1 && changer != -1) break;
            }

            const float bi = betas[changer];
            const float bj = betas[partner];
            const float ei = all_energies[changer];
            const float ej = all_energies[partner];
            const float dE = std::exp((-bj * ei - bi * ej) / (-bi * ei - bj * ej));
            if ((int)(dE > std::rand() / (float)(RAND_MAX))) {
              const auto tmp = betas[changer];
              betas[changer] = betas[partner];
              betas[partner] = tmp;
              const auto tmp_idx = betas_idx[changer];
              betas_idx[changer] = betas_idx[partner];
              betas_idx[partner] = tmp_idx; 
            }
          }
      }
      else
      {
          // (1,2), (3,4) ...
          for(int i=1;i<sz-1;i+=2)
          {
            int changer, partner; changer = partner = -1;
            for (int j=0;j<sz;++j)
            {
                if (betas_idx[j] == i) changer = j;
                if (betas_idx[j] == i + 1) partner = j;
                if (partner != -1 && changer != -1) break;
            }

            const float bi = betas[changer];
            const float bj = betas[partner];
            const float ei = all_energies[changer];
            const float ej = all_energies[partner];
            const float dE = std::exp((-bj * ei - bi * ej) / (-bi * ei - bj * ej));
            if ((int)(dE > std::rand() / (float)(RAND_MAX))) {
              const auto tmp = betas[changer];
              betas[changer] = betas[partner];
              betas[partner] = tmp;
              const auto tmp_idx = betas_idx[changer];
              betas_idx[changer] = betas_idx[partner];
              betas_idx[partner] = tmp_idx; 
            }
          }
      }
  }

  MPI_Bcast(betas, sz, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(betas_idx, sz, MPI_INT, 0, MPI_COMM_WORLD);
}



void try_exchange(short *lattice, const int &L, short *copies,
                  float *all_energies, float *betas, int *betas_idx) {
  int rk, sz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);
  float rank_energy = energy(lattice, L);
  MPI_Gather(&rank_energy, 1, MPI_FLOAT, all_energies, 1, MPI_FLOAT, 0,
             MPI_COMM_WORLD);
  if (rk == 0) {
    int changer = betas_idx[std::rand() % sz];
    int partner = betas_idx[std::rand() % sz];
    while (std::abs(changer - partner) != 1) // want to get a neighboring temperature
      partner = betas_idx[std::rand() % sz];

    const float bi = betas[changer];
    const float bj = betas[partner];
    const float ei = all_energies[changer];
    const float ej = all_energies[partner];
    const float dE = std::exp((-bj * ei - bi * ej) / (-bi * ei - bj * ej));
    if ((int)(dE > std::rand() / (float)(RAND_MAX)))
    {
      const auto tmp = betas[changer];
      betas[changer] = betas[partner];
      betas[partner] = tmp;
      const auto tmp_idx = betas_idx[changer];
      betas_idx[changer] = betas_idx[partner];
      betas_idx[partner] = tmp_idx;
    }
  }

  MPI_Bcast(betas, sz, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(betas_idx, sz, MPI_INT, 0, MPI_COMM_WORLD);
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

  assert(sz == 8);

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

  // create and distribute lattice to all ranks -- much better to just have
  // random initial configs for all ranks!
  //
  // if (rk == 0)
  build_lattice(lattice, L, r, b);

  // MPI_Bcast(lattice, L * L * L, MPI_SHORT, 0, MPI_COMM_WORLD);

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

  short *copies = nullptr;
  if (rk == 0) {
    copies = (short *)malloc(sizeof(short) * L * L * L * sz);
  }

  const int nsweeps = (1 << 14) + 1;

  float min_beta = .3;
  float max_beta = 4.;
  float *betas = (float *)malloc(sizeof(float) * sz);
  float t = min_beta;
  int currbeta = 0;
  // ATTENTION: this here is hardcoded for 8 (!!!!) ranks -- more or less ranks (ie. sz != 8) would yield a bug
  if (sz != 8) return 1;
  while (t <= max_beta) {
    betas[currbeta++] = t;
    t *= 1.4;
  }

  int curr = 1<<10;

  int curr_print = 1;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> uni(0., 1.);
  assert(L % 4 == 0 && "L must be a factor of 4!");

  float *all_energies = nullptr;
  int *beta_idx = (int *)malloc(sizeof(int) * sz);
  if (rk == 0) {
    all_energies = (float *)malloc(sizeof(float) * sz);
  }
  for (int i = 0; i < sz; ++i)
    beta_idx[i] = i;

  std::vector<std::vector<float>> betas_saved;

  float mybeta = betas[rk];
  for (int i = 1; i <= nsweeps; ++i) {
    mybeta = betas[rk];
    sweep_fully_random(lattice, N, mybeta, thread_generators, nn_dis, unis);
    //sweep(lattice, N, thread_generators, nn_dis, unis, mybeta, slice_coords);
#pragma omp master
    {
      if (i % 10 == 0)
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
        // change it back if criterion is not fulfilled
        if (uni(gen) >= std::exp(-mybeta * dE)) {
          exchange(lattice, N, red, blue);
        }
      }

      if (i == curr && sz > 1) {
        // get all configurations into `copies`
        MPI_Gather(lattice, L * L * L, MPI_SHORT, copies, L * L * L, MPI_SHORT,
                   0, MPI_COMM_WORLD);
        // try to exchange configurations between neighboring temperatures
        // -> we actually just exchange betas[i] and betas[j] (if successful!)
        // HERE
        try_exchange(lattice, L, copies, all_energies, betas, beta_idx);
        curr += 1000;
      }

      if (i == curr_print) {
        std::vector<float> betas_copy;
        betas_copy.push_back(i);
        for (int i = 0; i < sz; ++i)
          betas_copy.push_back(betas[i]);
        betas_saved.push_back(betas_copy);
        MPI_Gather(lattice, L * L * L, MPI_SHORT, copies, L * L * L, MPI_SHORT,
                   0, MPI_COMM_WORLD);

        //if (rk == 0)
        //{
        //  std::cout << i << ",";
        //  for(int i=0;i<sz;++i)
        //  {
        //      if (i < sz-1)
        //          std::cout << energy(copies + i*L*L*L, L) << ",";
        //      else
        //          std::cout << energy(copies + i*L*L*L, L) << "\n";
        //  }
        //}



        if (rk == 0) {
          dump_copies(copies, L);
        }
        curr_print *= 2;
      }
    }
  }

  if (rk == 0) {
    for (const auto &b : betas_saved) {
      // one extra for epoch
      for (int i = 1; i < sz + 1; ++i)
        std::cout << b[i] << " ";
      std::cout << "\n";
    }
  }

#pragma omp master
  {
  free(betas);
  free(beta_idx);
  if (rk == 0) {
    free(all_energies);
  }
  free(lattice);
  }
  MPI_Finalize();
  return 0;
}
