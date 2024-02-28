#include "maps.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

void build_lattice_diag(const int &num_particles, const int &num_red,
                        const int &num_blue, std::mt19937 &generator,
                        std::uniform_int_distribution<> &indices) {
  for (int i = 0; i < L * L * L; ++i)
    set_value_lattice(i, 0);
  int curr_red, curr_blue;
  curr_red = curr_blue = 0;
  while (curr_red < num_red) {
    const int site = indices(generator);
    const auto i = revert_table[3 * site + 0];
    const auto j = revert_table[3 * site + 1];
    const auto k = revert_table[3 * site + 2];
    if (((i + j + k) & 1) == 0 &&
        static_cast<int>(get_value_lattice(site)) == 0) {
      set_value_lattice(site, 1);
      curr_red++;
      continue;
    }
  }
  while (curr_blue < num_blue) {
    const int site = indices(generator);
    const auto i = revert_table[3 * site + 0];
    const auto j = revert_table[3 * site + 1];
    const auto k = revert_table[3 * site + 2];
    if (((i + j + k) & 1) == 1 &&
        static_cast<int>(get_value_lattice(site)) == 0) {
      set_value_lattice(site, 2);
      curr_blue++;
    }
  }
  int nred, nblue;
  nred = nblue = 0;
  for (int i = 0; i < L * L * L; ++i) {
    if (get_value_lattice(i) == 1)
      nred++;
    else if (get_value_lattice(i) == 2)
      nblue++;
  }
  assert(nred == num_red);
  assert(nblue == num_blue);
}

void exchange(const int &site, const int &to) {
  const auto tmp = get_value_lattice(site);
  set_value_lattice(site, get_value_lattice(to));
  set_value_lattice(to, tmp);
}

#define ONETWO(x) ((x & 0x1) || (x & 0x2))
const float local_energy_packed(const int &site) {
  const int ty = get_value_lattice(site);
  if (ty == 0)
    return 0.;
  const float connection = ty == 1 ? 3. : 5.;
  const int *nne = &(nearest_neighbors[NUM_NN * site]);
  const int e =
      ONETWO(get_value_lattice(nne[0])) + ONETWO(get_value_lattice(nne[1])) +
      ONETWO(get_value_lattice(nne[2])) + ONETWO(get_value_lattice(nne[3])) +
      ONETWO(get_value_lattice(nne[4])) + ONETWO(get_value_lattice(nne[5]));
  float current = static_cast<int>(e) - connection;
  return current * current;
}

const float nn_energy_packed(const int &site) {
  float res = local_energy_packed(site);
  const int *nne = &(nearest_neighbors[NUM_NN * site]);
  for (int i = 0; i < NUM_NN; ++i)
    res += local_energy_packed(nne[i]);
  return res;
}

void nonlocal_sweep(const float &beta, std::mt19937 &generator,
                    std::uniform_int_distribution<> &indices,
                    std::uniform_real_distribution<> &uni) {
  for (int i = 0; i < L * L * L; ++i) {
    const int site = indices(generator);
    const auto mv = indices(generator);
    if (get_value_lattice(site) == get_value_lattice(mv))
      continue;
    const float E1 = nn_energy_packed(site) + nn_energy_packed(mv);
    exchange(site, mv);
    const float E2 = nn_energy_packed(site) + nn_energy_packed(mv);
    const float dE = E2 - E1;
    if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
      continue;
    exchange(site, mv);
  }
}

void local_sweep(const float &beta, std::mt19937 &generator,
                 std::uniform_int_distribution<> &indices,
                 std::uniform_real_distribution<> &uni) {
  for (int i = 0; i < L * L * L; ++i) {
    const int site = indices(generator);
    const auto nb = indices(generator) % NUM_NN;
    const auto mv = nearest_neighbors[NUM_NN * site + nb];
    if (get_value_lattice(site) == get_value_lattice(mv))
      continue;
    const float E1 = nn_energy_packed(site) + nn_energy_packed(mv);
    exchange(site, mv);
    const float E2 = nn_energy_packed(site) + nn_energy_packed(mv);
    const float dE = E2 - E1;
    if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
      continue;
    exchange(site, mv);
  }
}

int energy() {
  int e = 0;
  for (int i = 0; i < L * L * L; ++i)
    e += local_energy_packed(i);
  return e;
}

bool check_cpy(uint8_t *lat) {

  for (int i = 0; i < L * L * L / 4; ++i)
    if (packed_lattice[i] != lat[i])
      return false;
  return true;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "run as: ./bin beta\n";
    return 1;
  }
  const auto arg1 = argv[1];  
  const auto beta = atof(arg1);
  

  const float rho = .75;
  // convenience defs, it's actually rho1 = .6 and rho2 = .4
  // ie. here rho1_used = rho1 * rho
  const float rho1 = .45;
  // const float rho2 = .3;
  const int N = (int)(lat_size * rho);
  const int N1 = (int)(rho1 * lat_size);
  const int N2 = N - N1;
  const auto max_idx = L * L * L - 1;
  auto uni = std::uniform_real_distribution<>(0., 1.);
  auto indices = std::uniform_int_distribution<>(0, max_idx);
  auto generator = std::mt19937();
  generator.seed(__rdtsc());
  generate_tables();

  build_lattice_diag(N, N1, N2, generator, indices);
    
  
  int burnin = 1;
  if (beta <= 2.5) burnin = 1 << 16;
  else if (beta <= 3.5) burnin = 1 << 21;
  else burnin = 1 << 25;
    
  for (int i = 0; i < burnin; ++i) {
    nonlocal_sweep(beta, generator, indices, uni); 
  }

  const int max_collect = 10000;  
  uint8_t  *config_collection = (uint8_t *)malloc(sizeof(uint8_t) * max_collect * L * L * L / 4); 
  const int config_size = L * L * L / 4;
  std::copy(packed_lattice, packed_lattice + config_size, config_collection); 

  for(int i = 0; i < max_collect; ++i)
  {
    std::copy(packed_lattice, packed_lattice + config_size, config_collection + i * L * L * L / 4);
    nonlocal_sweep(beta, generator, indices, uni);
  }

  for(int i = 0; i < max_collect; ++i)
  {
      auto lattice_ptr = config_collection + i * L * L * L / 4;
      for(int s = 0; s < L * L *L; ++s)
          std::cout << static_cast<int>(get_value(lattice_ptr, s)) << " ";
      std::cout << "\n";
  } 
 
  free(config_collection);
  return 0;
}
