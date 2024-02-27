#include <array>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>



#define L 20
constexpr int lat_size = L * L * L;
constexpr int lat_size_packed = L * L * L / 4;
uint8_t  packed_lattice[lat_size_packed];
int  nearest_neighbors[lat_size * 6];
int  revert_table[lat_size * 3];
int  revert_nn_table[lat_size * 3 * 6];
int  forward_table[L][L][L];
float local_energies[L * L * L];
int local_e_update_site[2][7];
float local_e_update_ey[2][7]; 

uint8_t get_value_lattice(const int &i, const int &j, const int &k)
{ 
    const int idx_in = (k + L * (j + i * L));
    //const int idx    = idx_in / 4; 
    //const int offset = (idx_in % 4) * 2;
    const int idx    = idx_in >> 2; 
    const int offset = (idx_in & 3) * 2;
    return (packed_lattice[idx] >> offset) & 3;
}

uint8_t get_value_lattice(const int & idx_in)
{ 
    //const int idx = idx_in / 4; 
    //const int offset = (idx_in % 4) * 2;  
    const int idx    = idx_in >> 2; 
    const int offset = (idx_in & 3) * 2;
    return (packed_lattice[idx] >> offset) & 3;
}

void set_value_lattice(const int &i, const int &j, const int &k, const uint8_t &value)
{ 
    const int idx_in = (k + L * (j + i * L));
    //const int idx    = idx_in / 4;
    //const int offset = (idx_in % 4) * 2;
    const int idx    = idx_in >> 2; 
    const int offset = (idx_in & 3) * 2;
    packed_lattice[idx] &= ~(3 << offset);
    packed_lattice[idx] |= ((value & 3) << offset);
}

void set_value_lattice(const int &idx_in, const uint8_t &value)
{
    //const int idx = idx_in / 4; 
    //const int offset = (idx_in % 4) * 2;
    const int idx    = idx_in >> 2; 
    const int offset = (idx_in & 3) * 2;
    packed_lattice[idx] &= ~(3 << offset);
    packed_lattice[idx] |= ( (value & 3) << offset);
}


void build_nn_list() {
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k) {
        const auto site = k + L * (j + i * L);
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
        nearest_neighbors[6 * site + 0] = f; 
        nearest_neighbors[6 * site + 1] = b;
        nearest_neighbors[6 * site + 2] = u; 
        nearest_neighbors[6 * site + 3] = d;
        nearest_neighbors[6 * site + 4] = l; 
        nearest_neighbors[6 * site + 5] = r;
      }
}


void generate_revert_table() {
  for (int i = 0; i < L; ++i)
  {
    for (int j = 0; j < L; ++j)
    {
      for (int k = 0; k < L; ++k) {
        const auto site = k + L * (j + i * L);
        revert_table[3 * site + 0] = i;
        revert_table[3 * site + 1] = j;
        revert_table[3 * site + 2] = k;
      }
    }
  }
}

std::array<std::array<int, 3>, 6>
get_neighbors_triplet(const int &i, const int &j, const int &k) {
  int ii, kk, jj;
  ii = (i == 0 ? L - 1 : i - 1);
  jj = (j == 0 ? L - 1 : j - 1);
  kk = (k == 0 ? L - 1 : k - 1);
  std::array<int, 3> f = {ii % L, j, k};
  std::array<int, 3> b = {(i + 1) % L, j, k};
  std::array<int, 3> u = {i, jj % L, k};
  std::array<int, 3> d = {i, (j + 1) % L, k};
  std::array<int, 3> l = {i, j, kk % L};
  std::array<int, 3> r = {i, j, (k + 1) % L};
  return {u, d, l, r, f, b};
}

void generate_neighbor_revert_table() { 
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k) {
        const auto site = k + L * (j + i * L);
        const auto n = get_neighbors_triplet(i, j, k);
        int counter = 0;
        for (const auto & ni : n)
        {
            for(int nj = 0; nj < 3; ++nj)
            {
                revert_nn_table[3 * 6 * site + 3 * counter + nj] = ni[nj];
            }
            counter += 1;
        }
      }
}

void build_forward_table()
{
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k)
       forward_table[i][j][k] = k + L * (j + i*L); 
}

void build_energies()
{
    for(int i=0;i<L*L*L;++i)
        local_energies[i] = 0.;
}

void generate_tables()
{
    build_nn_list();
    generate_revert_table();
    generate_neighbor_revert_table();
    build_forward_table();
    build_energies();
}
