#include <array>
#include <iostream>
#include <stdlib.h>
#include <unordered_map>

// fixing number of threads -- adheres to cluster + works locally
#define NUM_THREADS 4

// global objects defining the lattice graph with periodic boundary conditions
// using lookup-tables to reduce cycles needed in the sweeps

// global object per rank definining nearest neighbors as indices [0, L^3) ->
// ([0, L^3))^6
std::unordered_map<int, std::array<int, 6>> nn[NUM_THREADS];
// global object per rank definining mapping from index to triplet [0, L^3) ->
// [0, L) x [0, L) x [0, L)
std::unordered_map<int, std::array<int, 3>> revert_table[NUM_THREADS];
// global object per rank defining reverted nearest neighbors [0, L^3) -> ([0,
// L) x [0, L) x [0, L))^6
std::unordered_map<int, std::array<std::array<int, 3>, 6>>
    revert_neighbor_table[NUM_THREADS];

void dump_slice(const int *__restrict lattice, const int L, const int i) {
  for (int j = 0; j < L; ++j) {
    for (int k = 0; k < L; ++k)
      std::cout << lattice[k + L * (j + i * L)] << ",";
    std::cout << "\n";
  }
}

void dump_lattice(const int *__restrict lattice, int L) {
  for (int i = 0; i < L; ++i) {
    dump_slice(lattice, L, i);
    std::cout << "\n";
  }
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

std::unordered_map<int, std::array<int, 6>> get_nn_list(const int &L) {
  std::unordered_map<int, std::array<int, 6>> neighbor_map;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k) {
        const auto site = k + L * (j + i * L);
        neighbor_map[site] = get_neighbors(i, j, k, L);
      }
  return neighbor_map;
}

static inline std::tuple<int, int, int> revert(int s, int L = 30) {
  const auto k = s % L;
  const auto j = ((s - k) / L) % L;
  const auto i = (s - k - j * L) / (L * L);
  return {i, j, k};
}

std::unordered_map<int, std::array<int, 3>>
generate_revert_table(const int &L) {
  std::unordered_map<int, std::array<int, 3>> revert_table;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k) {
        const auto site = k + L * (j + i * L);
        revert_table[site] = {i, j, k};
      }
  return revert_table;
}

std::array<std::array<int, 3>, 6>
get_neighbors_triplet(const int &i, const int &j, const int &k, const int &L) {
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

std::unordered_map<int, std::array<std::array<int, 3>, 6>>
generate_neighbor_revert_table(const int &L) {
  std::unordered_map<int, std::array<std::array<int, 3>, 6>>
      revert_neighbor_table;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k) {
        const auto site = k + L * (j + i * L);
        const auto n = get_neighbors_triplet(i, j, k, L);
        revert_neighbor_table[site] = n;
      }
  return revert_neighbor_table;
}
