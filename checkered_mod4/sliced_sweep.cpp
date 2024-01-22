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

void dump_slice(const short *__restrict lattice, const int N, const int i) {
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k)
      std::cout << lattice[k + N * (j + i * N)] << ",";
    std::cout << "\n";
  }
}

void dump_lattice(const short *__restrict lattice, int N) {
  for (int i = 0; i < N; ++i)
  {
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

static inline std::tuple<int, int, int> revert(int s, int N = 30) {
  const auto k = s % N;
  const auto j = ((s - k) / N) % N;
  const auto i = (s - k - j * N) / (N * N);
  return {i, j, k};
}

std::array<int, 6> get_neighbors(const int &i, const int &j, const int &k,
                                 const int &N) {
  // THIS right here: NOT REVERSIBLE -> we have overlapping neighbors for some sites :( bad.
  //if (i == 0) {
  //  u = k + N * (j + (N - 1) * N);
  //}
  //if (i == N - 1) {
  //  d = k + N * j;
  //}
  //if (j == 0) {
  //  l = k + N * ((N - 1) + i * N);
  //}
  //if (j == N - 1) {
  //  r = k + N * (i * N);
  //}
  //if (k == 0) {
  //  f = N - 1 + N * (j + i * N);
  //}
  //if (k == N - 1) {
  //  b = N * (j + i * N);
  //}
  //if (i > 0) {
  //  u = k + N * (j + (i - 1) * N);
  //}
  //if (i < N - 1) {
  //  d = k + N * (j + (i + 1) * N);
  //}
  //if (j > 0) {
  //  l = k + N * (j - 1 + i * N);
  //}
  //if (j < N - 1) {
  //  r = k + N * (j + 1 + i * N);
  //}
  //if (k > 0) {
  //  f = k - 1 + N * (j + i * N);
  //}
  //if (k < N - 1) {
  //  b = k + 1 + N * (j + i * N);
  //}
  int ii, kk, jj;
  ii = (i==0? N-1 : i-1);
  jj = (j==0? N-1 : j-1);
  kk = (k==0? N-1 : k-1);
  const int f = k + N * ( j + (ii % N)* N); 
  const int b = k + N * ( j + ((i+1) % N)* N);
  const int u = k + N * ((jj % N) + i* N); 
  const int d = k + N * (((j+1) % N) + i* N);
  const int l = (kk % N) + N*(j + i *N); 
  const int r = ((k+1) % N) + N*(j + i *N); 
  //assert(f >= 0 && b >= 0);
  //assert(u >= 0 && d >= 0);
  //assert(l >= 0 && r >= 0);
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
  if (grid[site] == 0) return 0.;

  const float connection = grid[site] == 1 ? 3. : 5.; 
  const auto [l, r, u, d, f, b] = nn[site];
  float current =
      static_cast<float>(ONETWO(grid[u]) + ONETWO(grid[d]) + ONETWO(grid[l]) +
                         ONETWO(grid[r]) + ONETWO(grid[f]) + ONETWO(grid[b]));
  current = current - connection;
  return current * current;
}

static inline const float nn_energy(short *__restrict lattice, const int N, const int &i, const int &j, const int &k)
{
  const auto site = k + N * (j + N * i);
  if (lattice[site] == 0) return 0.;
  float res = local_energy(lattice, N, i, j, k);
//#pragma unroll(6)
  for(const auto & e : nn[site]) {
      const auto [m, l, n] = revert(e, N);
      res += local_energy(lattice, N, m, l, n);
  }
  return res;
}


// what works: shifted + scaled dE
const float scale = 750;// 750.; // the energy difference has range [-750,750] -- somewhat Gaussian but with gaps! -- truncated the Gaussian for now
const float A = 5.;
static inline void step(short *__restrict lattice, const int N, const int &i, const int &j, const int &k,
                 std::vector<std::mt19937> &gens,
                 std::vector<std::uniform_int_distribution<>> &nn_dis,
                 std::vector<std::uniform_real_distribution<>> &unis,
                 const float beta)
{
    const auto t = omp_get_thread_num();
    const auto nthreads = omp_get_num_threads();
    const auto site = k + N * (j + i * N);
    if (lattice[site] == 0) return;

    const auto nn = ::nn[site];
    const auto to_mv = nn_dis[t](gens[t]);

    const auto mv = nn[to_mv];
    if (!(lattice[site] > 0 && lattice[mv] == 0)) return;
    auto [mi, mj, mk] = revert(mv, N);

    const volatile float E1 = nn_energy(lattice, N, i, j, k) + nn_energy(lattice, N, mi, mj, mk);// local_energy(lattice, N, i, j, k) + local_energy(lattice, N, mi, mj, mk);
    exchange(lattice, N, site, mv);

    const volatile float E2 = nn_energy(lattice, N, i, j, k) + nn_energy(lattice, N, mi, mj, mk);// local_energy(lattice, N, i, j, k) + local_energy(lattice, N, mi, mj, mk);
    
    // * this version is apparently used in the PLR paper? (works in numpy when using abs val)
    const auto dE = (E2-E1);
    const auto crit = (std::exp(-beta * dE) < 1.? std::exp(-beta * dE) : 1.);
//#pragma omp critical
//    if (dE <= 0)
//      std::cout << site << "," << mv << "," << E2 - E1 << "\n";
    
    
    
    
    
    // THE IMPLEMENTATION WE?RE CURRENTLY WORKING WITH!
    //const auto dE = (E2-E1+scale) / 2 / scale; 
    ////const auto dE = (E2-E1) / scale;
    //const auto w0 = 1.;
    //// Metropolis criterion
    //const auto crit = w0 * (std::exp(-beta * dE * A) < 1.? std::exp(-beta * dE * A) : 1.); 
    //
    //// Glauber dynamics
    ////const auto crit = w0 * ( .5 * (1. - std::tanh(std::exp(-(dE) * beta))));

    auto dist = mv - site;
    
//#pragma omp critical
//    std::cout << dist << "," << E2-E1 << "," << to_mv << "\n";

    if (unis[t](gens[nthreads + t]) < crit) {
      // accept move
//#pragma omp critical
//        std::cout << site << "," << mv << "," << E2 - E1 << "\n";
      return;
    }
    else {
      exchange(lattice, N, site, mv);
    }    
}


void slice_sweep(short *__restrict lattice, const int N, const int &i,
                 std::vector<std::mt19937> &gens,
                 std::vector<std::uniform_int_distribution<>> &nn_dis,
                 std::vector<std::uniform_real_distribution<>> &unis,
                 const float beta) {
    {
      // uniformly traverse slice of the lattice
      const auto stride = 4;
      for(int s = 0; s < stride; ++s)
        for(int t = 0; t < stride; ++t)
          for (int j = s; j < N; j+=stride) {
            for (int k = t; k < N; k+=stride) {
              step(lattice, N, i, j, k, gens, nn_dis, unis, beta);
            }
          }
    }
}


void slice_sweep_randomized(short *__restrict lattice, const int N, const int &i,
                 std::vector<std::mt19937> &gens,
                 std::vector<std::uniform_int_distribution<>> &nn_dis,
                 std::vector<std::uniform_real_distribution<>> &unis,
                 std::vector<std::vector<std::tuple<int, int>>> &slice_coords,
                 const float beta)
{
    const auto t = omp_get_thread_num();
    {
      const auto tid = omp_get_thread_num();
      std::shuffle(slice_coords[tid].begin(), slice_coords[tid].end(), gens[tid]);
      for(const auto & [j, k] : slice_coords[tid])
      {
//#pragma omp critical
//          std::cout << t << "," << i << "," << j << "," << k << "\n";
          step(lattice, N, i, j, k, gens, nn_dis, unis, beta); 
      }
    }
}

void sweep(short *__restrict lattice, const int N,
           std::vector<std::mt19937> &gens,
           std::vector<std::uniform_int_distribution<>> &nn_dis,
           std::vector<std::uniform_real_distribution<>> &unis,
           const float beta, std::vector<std::vector<std::tuple<int, int>>> &slice_coords) {
//#pragma omp parallel
    {
      const int stride = 4;
      const int mod = N % stride;
      assert( mod == 0);

      

      std::vector<int> idx = {0, 1, 2, 3};
      int ii = 0;
#pragma omp master
      std::shuffle(idx.begin(), idx.end(), gens[0]);

#pragma omp parallel shared(idx)
      for (int s = 0; s < stride; ++s) {
#pragma omp barrier // everyone needs acces to same s
        ii = idx[s];
        //ii = s;
#pragma omp for collapse(1) schedule(auto)
        for (int i = ii; i < N; i += stride) {
          //slice_sweep(lattice, N, i, gens, nn_dis, unis, beta);
          slice_sweep_randomized(lattice, N, i, gens, nn_dis, unis, slice_coords, beta);
        }
      }
      
    }
}

float energy(short *__restrict lattice, const int &N) {
  float e = 0.;
#pragma omp parallel for collapse(3) reduction(+ : e)
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        e += local_energy(lattice, N, i, j, k);
  return  e;
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

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "run as ./bin beta num_threads L swap_flag\n";
    return 1;
  }
  const auto arg_beta = argv[1];
  const auto beta = atof(arg_beta);
  const auto arg_nsw = argv[2];
  const auto nsw = atoi(arg_nsw);
  const auto arg_n = argv[3];
  const auto L = atoi(arg_n);
  const auto swap_flag = argv[4];
  const auto swap = atoi(swap_flag);

  const int N = L;
  const float rho = .75;
  const float rho1 = .4 * rho;
  const float rho2 = rho - rho1;
  const int numparts = static_cast<int>(rho * N * N * N);

  const int r = static_cast<int>(rho1 * N * N * N);
  const int b = numparts - r;

  assert(rho1 + rho2 == rho);

  if (L < 7)
    omp_set_num_threads(1);
  else if (L < 10)
    omp_set_num_threads(2);
  else
    omp_set_num_threads(nsw);
  ::nn = get_nn_list(N);
  //for(const auto [i, n] : ::nn)
  //{
  //    const auto [n1, n2, n3, n4, n5, n6] = n;
  //    std::cout << i << ": " << n1 << "," << n2 << "," << n3 << "," << n4 << "," << n5 << "," << n6 << "\n";
  //    for(int ii=0;ii<6;++ii)
  //        for(int j=0;j<6;++j)
  //            if (ii!=j && n[ii] == n[j]) std::cout << "site " << i << ", have double " << n[ii] << "\n";
  //}

  //if (beta == 5.) return 1;

  short *lattice = nullptr;
  short *lattice_back = nullptr;
  const unsigned int align = 64;
  const auto padded_N = (N * N * N + (align - 1)) & ~(align - 1);

  if (posix_memalign((void **)&lattice, align, padded_N * sizeof(short)) != 0)
    return 1;

  build_lattice(lattice, N, r, b);
  
  if (posix_memalign((void **)&lattice_back, align, padded_N * sizeof(short)) !=
      0)
    return 1;

  // copying starting lattice
  for (int i = 0; i < N * N * N; ++i)
    lattice_back[i] = lattice[i];

  const int nsweeps = (1 << 10) + 1;
  std::vector<std::tuple<int, int, int>> data;

  data.push_back({0, logand(lattice_back, lattice, N), 0});

  int curr = 1;

  const int nthreads = get_num_threads();
  std::vector<std::mt19937> thread_generators(2 * nthreads);

#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
#pragma omp critical
      { 
        thread_generators[tid].seed(std::random_device()() + tid);
        thread_generators[omp_get_num_threads() + tid].seed(std::random_device()() +
                                                            100 * tid);
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
      for(int j=0;j<L;++j)
          for(int k=0;k<L;++k)
#pragma omp critical
              slice_coords[tid].emplace_back(j, k); 

    std::shuffle(slice_coords[tid].begin(), slice_coords[tid].end(), thread_generators[tid]); 
    std::sort(slice_coords[tid].begin(), slice_coords[tid].end());
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> uni(0., 1.);
  assert (L % 4 == 0 && "L must be a factor of 4!");
  for (int i = 1; i <= nsweeps; ++i) {
    sweep(lattice, N, thread_generators, nn_dis, unis, beta, slice_coords);
//    if (swap) {
//#pragma omp master
//      {
//        auto red = sample(lattice, N * N * N, r, 1);
//        auto blue = sample(lattice, N * N * N, b, 2);
//
//        auto [ii, j, k] = revert(red, N);
//        auto [x, y, z] = revert(blue, N);
//
//        float E1 = nn_energy(lattice, N, ii, j, k) +
//                   nn_energy(lattice, N, x, y, z);
//        exchange(lattice, N, red, blue);
//        float E2 = nn_energy(lattice, N, ii, j, k) +
//                   nn_energy(lattice, N, x, y, z);
//        auto dE = std::abs(E2 - E1);
//        if (uni(gen) >= std::exp(-beta * dE)) {
//          exchange(lattice, N, red, blue);
//        }
//      }
//    }

    if (i == curr) {
      const auto anded = logand(lattice, lattice_back, N);
      data.push_back({i, anded, energy(lattice, N)});
      curr *= 2;
      if ((float)(anded) / (float)(numparts) <= .15)
        break; 
    }
  }

  for (const auto &[epoch, anded, e] : data)
    std::cout << epoch << "," << anded << "," << e <<  "\n";

  free(lattice);
  free(lattice_back);
  return 0;
}
