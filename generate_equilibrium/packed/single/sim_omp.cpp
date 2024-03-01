/*
 * COMPILE COMMAND
 *
 * clang++-17 -fno-strict-aliasing -ftree-vectorize -pedantic -ffast-math -march=native -O3 -Wall -Wunknown-pragmas -fopenmp  -lm -lstdc++ -std=c++17 sim_omp.cpp -o to_omp && GOMP_CPU_AFFINITY="0,2,8,10" ./to_omp 7.
 *
 */

#include "maps_omp.h"
#include "npy.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <tuple>

static inline std::tuple<int, int, int> revert(int s) {
  const auto k = s % L;
  const auto j = ((s - k) / L) % L;
  const auto i = (s - k - j * L) / (L * L);
  return {i, j, k};
}

void build_lattice_diag(const int &num_particles, const int &num_red,
                        const int &num_blue, std::mt19937 &generator,
                        std::uniform_int_distribution<> &indices,
                        const int & tid) {

  for (int i = 0; i < L * L * L; ++i)
    set_value_lattice(i, 0, tid);
  int curr_red, curr_blue;
  curr_red = curr_blue = 0;
  while (curr_red < num_red) {
    const int site = indices(generator);
    const auto [i, j, k] = revert(site);
    
    if (((i + j + k) & 1) == 0 &&
        static_cast<int>(get_value_lattice(site, tid)) == 0) {
      set_value_lattice(site, 1, tid);
      curr_red++;
      continue;
    }
  }
  while (curr_blue < num_blue) {
    const int site = indices(generator);
    const auto [i, j, k] = revert(site);
    if (((i + j + k) & 1) == 1 &&
        static_cast<int>(get_value_lattice(site, tid)) == 0) {
      set_value_lattice(site, 2, tid);
      curr_blue++;
    }
  }
  int nred, nblue;
  nred = nblue = 0;
  for (int i = 0; i < L * L * L; ++i) {
    if (get_value_lattice(i, tid) == 1)
      nred++;
    else if (get_value_lattice(i, tid) == 2)
      nblue++;
  }
  assert(nred == num_red);
  assert(nblue == num_blue);
}

void exchange(const int &site, const int &to, const int & tid) {
  const auto tmp = get_value_lattice(site, tid);
  set_value_lattice(site, get_value_lattice(to, tid), tid);
  set_value_lattice(to, tmp, tid);
}

#define ONETWO(x) ((x & 0x1) || (x & 0x2))
const float local_energy_packed(const int &site, const int * nearest_neighbors, const int & tid) {
  const int ty = get_value_lattice(site, tid);
  if (ty == 0)
    return 0.;
  const float connection = ty == 1 ? 3. : 5.;
  const int *nne = &(nearest_neighbors[NUM_NN * site]);
  const int e =
      ONETWO(get_value_lattice(nne[0], tid)) + ONETWO(get_value_lattice(nne[1], tid)) +
      ONETWO(get_value_lattice(nne[2], tid)) + ONETWO(get_value_lattice(nne[3], tid)) +
      ONETWO(get_value_lattice(nne[4], tid)) + ONETWO(get_value_lattice(nne[5], tid));
  float current = static_cast<float>(e) - connection;
  return current * current;
}

const float nn_energy_packed(const int &site, const int * nearest_neighbors, const int & tid) {
  float res = local_energy_packed(site, nearest_neighbors, tid);
  const int *nne = &(nearest_neighbors[NUM_NN * site]);
  for (int i = 0; i < NUM_NN; ++i)
    res += local_energy_packed(nne[i], nearest_neighbors, tid);
  return res;
}

void nonlocal_sweep(const int & num_trials, const float &beta, std::mt19937 &generator,
                    std::uniform_int_distribution<> &indices,
                    std::uniform_real_distribution<> &uni,
                    const int * nearest_neighbors, const int & tid) {
  for (int i = 0; i < num_trials; ++i) {
    const int site = indices(generator);
    const auto mv = indices(generator);
    if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
      continue;
    const float E1 = nn_energy_packed(site, nearest_neighbors, tid) + nn_energy_packed(mv, nearest_neighbors, tid);
    exchange(site, mv, tid);
    const float E2 = nn_energy_packed(site, nearest_neighbors, tid) + nn_energy_packed(mv, nearest_neighbors, tid);
    const float dE = E2 - E1;
    if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
      continue;
    exchange(site, mv, tid);
  }
}

void local_sweep(const float &beta, std::mt19937 &generator,
                 std::uniform_int_distribution<> &indices,
                 std::uniform_real_distribution<> &uni,
                 const int * nearest_neighbors, const int & tid) {
  for (int i = 0; i < L * L * L; ++i) {
    const int site = indices(generator);
    const auto nb = indices(generator) % NUM_NN;
    const auto mv = nearest_neighbors[NUM_NN * site + nb];
    if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
      continue;
    const float E1 = nn_energy_packed(site, nearest_neighbors, tid) + nn_energy_packed(mv, nearest_neighbors, tid);
    exchange(site, mv, tid);
    const float E2 = nn_energy_packed(site, nearest_neighbors, tid) + nn_energy_packed(mv, nearest_neighbors, tid);
    const float dE = E2 - E1;
    if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
      continue;
    exchange(site, mv, tid);
  }
}


int energy(const int * nearest_neighbors,const int & tid) {
  int e = 0;
  for (int i = 0; i < L * L * L; ++i)
    e += local_energy_packed(i, nearest_neighbors, tid);
  return e;
}


int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "run as: ./bin beta\n";
    return 1;
  }
  const auto arg1 = argv[1];  
  const auto arg2 = argv[2];
  const auto beta = atof(arg1);
  const auto fname = std::string(arg2);

#pragma omp parallel
  assert(omp_get_num_threads() == NUM_THREADS);

  

  const float rho = .75;
  // convenience defs, it's actually rho1 = .6 and rho2 = .4
  // ie. here rho1_used = rho1 * rho
  const float rho1 = .45;
  // const float rho2 = .3;
  const int N = (int)(lat_size * rho);
  const int N1 = (int)(rho1 * lat_size);
  const int N2 = N - N1;
  const auto max_idx = L * L * L - 1;
  generate_tables();
  
  std::vector<std::tuple<int, double>> timing;
#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
    auto uni = std::uniform_real_distribution<>(0., 1.);
    auto indices = std::uniform_int_distribution<>(0, max_idx);
    auto generator = std::mt19937();
    generator.seed(__rdtsc() + tid * tid);

    //uint8_t * my_lattice = thread_lattice[tid];
    int * my_nn          = thread_nn[tid];
        

    build_lattice_diag(N, N1, N2, generator, indices, tid); 
    int burnin = 1000;
    //// > 10^4
    //if (beta <= 2.5) burnin = 1 << 16;
    //// > 10^6
    //else if (beta <= 3.5) burnin = 1 << 21;
    //// > 10^7
    //else burnin = 1 << 25;

    auto t = -omp_get_wtime();
    for (int i = 0; i < burnin; ++i) {
      nonlocal_sweep(L * L * L, beta, generator, indices, uni, my_nn, tid); 
    }
    t += omp_get_wtime();
#pragma omp critical
    timing.push_back({tid, t});
  }
#pragma omp barrier  

  constexpr int power2 = 10;
  constexpr int nlocal_sweeps = 1 << power2;
  // how many configs of this sweep (evenly spaced!) do we want to keep?
  
  // how many configs to save per thread
  constexpr int max_collect = 1 + power2 + 8; // always a little larger than what we actually collect to, again, avoid false sharing  
  
  
  uint8_t * config_collection = (uint8_t *)malloc(sizeof(uint8_t) * max_collect * packed_size * NUM_THREADS);  
  // easily control what's happening to memory allocated for the config copies
  for(int c = 0; c < max_collect * packed_size * NUM_THREADS; ++c)
      config_collection[c] = static_cast<uint8_t>(0x0);

  
// finish all memory operations in all threads before continuing
#pragma omp flush
  
  std::array< double, NUM_THREADS> times = {0.0};
  
#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
    const auto offset = (tid * max_collect) * packed_size;
    uint8_t * copy_spot  = config_collection + offset; 
    const uint8_t * my_lattice = thread_lattice[tid];
    std::copy(my_lattice, my_lattice + packed_size, copy_spot);
#pragma omp barrier
  }

#pragma omp parallel
  for (int d = 0; d < power2; ++d)
  { 
    {
#pragma omp flush 
      const auto tid = omp_get_thread_num();
      auto uni = std::uniform_real_distribution<>(0., 1.);
      auto indices = std::uniform_int_distribution<>(0, max_idx);
      auto generator = std::mt19937();
      generator.seed(__rdtsc() + tid * tid);
      const uint8_t * my_lattice = thread_lattice[tid];
      const int * my_nn          = thread_nn[tid]; 
      auto t = -omp_get_wtime();
      for (int i = 0; i < 1 << d; ++i)
      {
        // single nonlocal exchange trial
        //nonlocal_sweep(1, beta, generator, indices, uni, my_nn, tid);
        local_sweep(beta, generator, indices, uni, my_nn, tid);
      } 
      t += omp_get_wtime();
      // not forgetting the initial config
      const auto offset = (tid * max_collect + d + 1) * packed_size;
      uint8_t * copy_spot  = config_collection + offset; 
      std::copy(my_lattice, my_lattice + packed_size, copy_spot);
#pragma omp critical
      times[tid] += t;

#pragma omp barrier
    }
  }
  
 for(size_t t = 0; t < NUM_THREADS; ++t)
  { 
      for(size_t d = 0; d < power2 + 1; ++d)
      {
        const auto offset = (t * max_collect + d) * packed_size;
        uint8_t * current_config = config_collection + offset;        
        int nred, nblue;
        nred = nblue = 0;
        for(int i=0;i<L*L*L;++i)
        {
            if (static_cast<short>(get_value(current_config, i)) == 1) nred++;
            if (static_cast<short>(get_value(current_config, i)) == 2) nblue++;
        }
        if (N1 != nred)  std::cout << "it's red on " << t << " epoch=" << d  << "\n";
        if (N2 != nblue) std::cout << "it's blue on " << t << " epoch=" << d << "\n";
        assert(N1 == nred);
        assert(N2 == nblue); 
      }
  }

  short serialized_configs[NUM_THREADS][power2 + 1][L*L*L]; 
  for(size_t t = 0; t < NUM_THREADS; ++t)
  {

      for(size_t d = 0; d < power2 + 1; ++d)
      {
        const auto offset = (t * max_collect + d) * packed_size;
        uint8_t * current_config = config_collection + offset; 
        for(size_t s = 0; s < L*L*L; ++s)
            serialized_configs[t][d][s] = static_cast<short>(get_value(current_config, s)); 
      } 
  }

  npy::npy_data_ptr<short> d;
  d.data_ptr = reinterpret_cast<const short *>( &serialized_configs);
  d.shape = {NUM_THREADS, (power2 + 1),  L, L, L};
  d.fortran_order = false;
  npy::write_npy(fname, d);


  //std::cout << "RUNTIME, local no PT: ";
  //for(size_t t = 0; t < NUM_THREADS; ++t)
  //    std::cout << times[t] << " ";
  //std::cout << "\n";

  free(config_collection);
  return 0;
}
