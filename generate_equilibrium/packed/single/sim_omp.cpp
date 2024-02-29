/*
 * COMPILE COMMAND
 *
 * clang++-17 -fno-strict-aliasing -ftree-vectorize -pedantic -ffast-math -march=native -O3 -Wall -Wunknown-pragmas -fopenmp  -lm -lstdc++ -std=c++17 sim_omp.cpp -o to_omp && GOMP_CPU_AFFINITY="0,2,8,10" ./to_omp 7.
 *
 */

#include "maps_omp.h"

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

void nonlocal_sweep(const float &beta, std::mt19937 &generator,
                    std::uniform_int_distribution<> &indices,
                    std::uniform_real_distribution<> &uni,
                    const int * nearest_neighbors, const int & tid) {
  for (int i = 0; i < L * L * L; ++i) {
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
  if (argc != 2) {
    std::cout << "run as: ./bin beta\n";
    return 1;
  }
  const auto arg1 = argv[1];  
  const auto beta = atof(arg1);

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
    int burnin = 100;
    //// > 10^4
    //if (beta <= 2.5) burnin = 1 << 16;
    //// > 10^6
    //else if (beta <= 3.5) burnin = 1 << 21;
    //// > 10^7
    //else burnin = 1 << 25;

    auto t = -omp_get_wtime();
    for (int i = 0; i < burnin; ++i) {
      nonlocal_sweep(beta, generator, indices, uni, my_nn, tid); 
    }
    t += omp_get_wtime();
#pragma omp critical
    timing.push_back({tid, t});
  }

  timing.clear();
  

  constexpr int nlocal_sweeps = 1 << 10;
  constexpr int divider = 1 << 8;
  // how many configs to save per thread
  constexpr int max_collect = divider + 8; // always a little larger than what we actually collect to, again, avoid false sharing  
  //uint8_t  *config_collection = (uint8_t *)malloc(sizeof(uint8_t) * max_collect * packed_size * NUM_THREADS); 
  
  uint8_t * config_collection = (uint8_t *)malloc(sizeof(uint8_t) * max_collect * packed_size * NUM_THREADS); 
  
  for(int c = 0; c < max_collect * packed_size; ++c)
      config_collection[c] = static_cast<uint8_t>(0x0);

  
// finish all memory operations in all threads before continuing
#pragma omp flush
  std::cout << "max_collect=" << max_collect <<  "\n"; 
  std::cout << "packed_size=" << packed_size << "\n";
  std::cout << max_collect * packed_size * NUM_THREADS << "\n";
  std::array< double, NUM_THREADS> times = {0.0};
  constexpr int dist = nlocal_sweeps / divider;
#pragma omp parallel
  for (int d = 0; d < divider; ++d)
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
      for (int i = 0; i < dist; ++i)
      {
        local_sweep(beta, generator, indices, uni, my_nn, tid);
      } 
      t += omp_get_wtime();
      const auto offset = tid * max_collect + d;
      uint8_t * copy_spot  = config_collection + offset * packed_size; 
      std::copy(my_lattice, my_lattice + packed_size, copy_spot);
  

#pragma omp critical
      times[tid] += t;

#pragma omp barrier
    }
  }
  

  /*
#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
    auto uni = std::uniform_real_distribution<>(0., 1.);
    auto indices = std::uniform_int_distribution<>(0, max_idx);
    auto generator = std::mt19937();
    generator.seed(__rdtsc() + tid * tid);
    uint8_t * my_lattice = thread_lattice[tid];
    int * my_nn          = thread_nn[tid];

    auto t = -omp_get_wtime();
    for (int i = 0; i < nlocal_sweeps; ++i) {
#pragma omp barrier
      if (i % 1000)
        std::copy(my_lattice, my_lattice + packed_size, config_collection + tid * max_collect * packed_size + i * packed_size);
      float beta = betas[tid];
      local_sweep(beta, generator, indices, uni, my_nn, tid); 

//      if (i % 10 == 0)
//      {
//          if( tid == 0)
//          {
//            const auto rand_idx = indices(generator) % NUM_THREADS;
//            int tid_trial = betas_idx[rand_idx];
//            int tid_next  = betas_idx[(rand_idx + 1) % NUM_THREADS];
//            while(std::abs(tid_trial - tid_next) != 1)
//                tid_next = betas_idx[indices(generator) % NUM_THREADS];
//            const float bi = betas[tid_trial];
//            const float bj = betas[tid_next];
//            const int ei = energy(my_nn, tid_trial);
//            const int ej = energy(my_nn, tid_next);
//            const float dT = bi - bj;
//            const float dE = ei - ej;
//            if (dE <= 0 || uni(generator) < std::exp(-dT * dE))
//            {  
//              betas[tid_trial] = bj;
//              betas[tid_next]  = bi;
//              betas_idx[tid_trial] = tid_next;
//              betas_idx[tid_next]  = tid_trial;  
//            } 
//          }
//      }
//
//      {
//        if (tid == 0)
//        {
//          std::vector<int> ordering;
//          for(int t = 0; t < NUM_THREADS; ++t)
//              ordering.push_back(betas_idx[t]);
//          beta_orderings.push_back(ordering);
//        }
//      }

#pragma omp flush
    }


    t += omp_get_wtime();
#pragma omp critical
    timing.push_back({tid, t});
#pragma omp flush
  }
  */

   

 
  // + 1 for initial config
  //for(size_t e = 0; e < nlocal_sweeps; ++e)
  //{
  //    auto current_ordering = beta_orderings[e];
  //    for (const auto & t : current_ordering)
  //    { 
  //        uint8_t * current_config = config_collection + t * max_collect * packed_size + e; 
  //        for(size_t s = 0; s < L*L*L; ++s)
  //            std::cout << static_cast<short>(get_value(current_config, s)) << " ";
  //        std::cout << "\n"; 
  //    }
  //}


  for(size_t t = 0; t < NUM_THREADS; ++t)
  {
      
      for(size_t d = 0; d < divider; ++d)
      {
        const auto offset = t * max_collect + d;
        uint8_t * current_config = config_collection + offset * packed_size;
        
        int nred, nblue;
        nred = nblue = 0;
        for(int i=0;i<L*L*L;++i)
        {
            if (static_cast<short>(get_value(current_config, i)) == 1) nred++;
            if (static_cast<short>(get_value(current_config, i)) == 2) nblue++;
        }
        //if (N1 != nred)  std::cout << "it's red on " << t << " epoch=" << d * (nlocal_sweeps/divider) << "\n";
        //if (N2 != nblue) std::cout << "it's blue on " << t << " epoch=" << d * (nlocal_sweeps/divider)<< "\n";

        assert(N1 == nred);
        assert(N2 == nblue);
        //for(size_t s = 0; s < L*L*L; ++s)
        //  std::cout << static_cast<short>(get_value(current_config, s)) << " ";
        //std::cout << "\n";  
      }
  }

  std::cout << "RUNTIME, local no PT: ";
  for(size_t t = 0; t < NUM_THREADS; ++t)
      std::cout << times[t] << " ";
  std::cout << "\n";

  free(config_collection);
  return 0;
}
