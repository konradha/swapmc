#include "maps.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
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
  const int *nne = &(nearest_neighbors[6 * site]);
  const int e =
      ONETWO(get_value_lattice(nne[0])) + ONETWO(get_value_lattice(nne[1])) +
      ONETWO(get_value_lattice(nne[2])) + ONETWO(get_value_lattice(nne[3])) +
      ONETWO(get_value_lattice(nne[4])) + ONETWO(get_value_lattice(nne[5]));
  float current = static_cast<int>(e) - connection;
  return current * current;
}

const float nn_energy_packed(const int &site) {
  float res = local_energy_packed(site);
  const int *nne = &(nearest_neighbors[6 * site]);
  for (int i = 0; i < 6; ++i)
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
    const auto nb = indices(generator) % 6;
    const auto mv = nearest_neighbors[6 * site + nb];
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
  if (argc != 5) {
    std::cout << "run as: ./bin beta nonlocal_power local_power path-to-output\n";
    return 1;
  }
  const auto arg1 = argv[1];
  const auto arg2 = argv[2];
  const auto arg3 = argv[3];
  const auto arg4 = argv[4];
  const auto beta = atof(arg1);
  const auto nonlocal_sweeps = atoi(arg2);
  const auto local_sweeps = atoi(arg3);
  const auto fname = std::string(arg4);

  

  const float rho = .75;
  // convenience defs
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
    
  
  
 
  
  for (int i = 1; i < (1 << nonlocal_sweeps) + 1; ++i) {
    nonlocal_sweep(beta, generator, indices, uni);
  }

  const int max_collect = 1 << 10; 
  uint8_t  *config_collection = (uint8_t *)malloc(sizeof(uint8_t) * max_collect * L * L * L / 4); 
  const int config_size = L * L * L / 4;
  std::copy(packed_lattice, packed_lattice + config_size, config_collection); 

  std::basic_filebuf<char> fb;
  fb.open(arg4, std::ios::out | std::ios::binary);
  
  std::ostream os(&fb);

  int collecter = 1;
  int collecter_repeat = 0;
  int checkin = 1;
  int multiplier = 0;
  std::vector<int> epochs;

  size_t s = 1;


  // can be done smartly
  // outer loop powers (10, 14) (14, 18) (18, 22) ... to ... %10 %100 %1000
  // inner loop: sweep and check at every %k to save config

  for(; s < (1 << local_sweeps) + 1 && s < (1 << 10) + 1; ++s)
  {
    local_sweep(beta, generator, indices, uni);
    const auto newpos = config_collection + collecter * config_size;
    std::copy(packed_lattice, packed_lattice + config_size, newpos);
    collecter += 1;
    epochs.push_back(s);
  }


  multiplier = 1<<14;
  checkin *= 10;
  for(; s < (1 << local_sweeps) + 1 && s < multiplier + 1; ++s)
  {

    local_sweep(beta, generator, indices, uni);
    if ( s % checkin == 0)
    {
      const auto newpos = config_collection + collecter * config_size;
      std::copy(packed_lattice, packed_lattice + config_size, newpos);
      collecter += 1;
      epochs.push_back(s);
    }

    if (collecter >= max_collect)
    {
      os.write((char *)config_collection, max_collect * L * L * L / 4);
      collecter = 0;
      collecter_repeat++;
    }
  }


  if (collecter > 0)
    os.write((char *) config_collection, collecter * L * L * L / 4); 
  fb.close(); 
  free(config_collection);

  for(int i=0;i<epochs.size();++i)
      std::cout << epochs[i] << " ";
  std::cout << "\n";


  return 0;
}
