#include "maps.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

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
  if (argc != 4) {
    std::cout << "run as: ./bin beta nonlocal_power local_power\n";
    return 1;
  }
  const auto arg1 = argv[1];
  const auto arg2 = argv[2];
  const auto arg3 = argv[3];
  const auto beta = atof(arg1);
  const auto nonlocal_sweeps = atoi(arg2);
  const auto local_sweeps = atoi(arg3);

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
  // for(int i=0;i<L*L*L;++i)
  //     std::cout << (int)get_value_lattice(i) << " ";
  // std::cout << "\n";

  uint8_t cpy_lat[L * L * L / 4];
  memcpy(cpy_lat, packed_lattice, L * L * L / 4);

  const int nsweeps = 1 << 10;
  const int max_collect = nsweeps;
  uint8_t config_collection[max_collect * L * L * L / 4];
  // copy initial config
  const int config_size = L * L * L / 4;
  std::copy(packed_lattice, packed_lattice + config_size, config_collection);
  int printer = 1;
  int collecter = 1;
  int collection_idx = 0;
  // std::cout << 0 << ", " << energy() << "\n";
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  for (int i = 1; i < (1 << nonlocal_sweeps) + 1; ++i) {
    nonlocal_sweep(beta, generator, indices, uni);
    // if(printer == i)
    //{
    //     std::cout << i << ", " << energy() << "\n";
    //     printer *= 2;
    // }
    // if (collecter == i)
    //{
    //     const auto newpos = config_collection + collection_idx * config_size;
    //     std::copy(packed_lattice, packed_lattice + config_size, newpos);
    //     collecter += 1;
    //     collection_idx += 1;
    // }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  // std::cout << "Duration: " << duration << " ms" << std::endl;

  for (int i = 1; i < (1 << local_sweeps) + 1; ++i) {
    local_sweep(beta, generator, indices, uni);
    // if(printer == i)
    //{
    //     std::cout << i << ", " << energy() << "\n";
    //     printer *= 2;
    // }
    if (collecter == i) {
      const auto newpos = config_collection + collection_idx * config_size;
      std::copy(packed_lattice, packed_lattice + config_size, newpos);
      collecter += 1;
      collection_idx += 1;
    }
  }

  for (int c = 0; c < collection_idx; ++c) {
    for (int i = 0; i < L * L * L; ++i)
      std::cout << static_cast<int>(
                       get_value(config_collection + c * config_size, i))
                << " ";
    std::cout << "\n";
  }

  return 0;
}
