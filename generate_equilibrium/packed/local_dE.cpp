#include <iostream>
#include <algorithm>
#include <vector>
#include <tuple>

#include <fstream>
#include <sstream>

#include <array>

#include <stdlib.h>

// global objects defining the lattice graph with periodic boundary conditions
// using lookup-tables to reduce cycles needed in the sweeps

// global object per rank definining nearest neighbors as indices [0, L^3) ->
// ([0, L^3))^6
std::unordered_map<int, std::array<int, 6>> nn;
// global object per rank definining mapping from index to triplet [0, L^3) ->
// [0, L) x [0, L) x [0, L)
std::unordered_map<int, std::array<int, 3>> revert_table;
// global object per rank defining reverted nearest neighbors [0, L^3) -> ([0,
// L) x [0, L) x [0, L))^6
std::unordered_map<int, std::array<std::array<int, 3>, 6>>
    revert_neighbor_table;

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

static inline void exchange(int *__restrict grid, const int &N, const int &site,
                            const int &to) {
  const auto tmp = grid[site];
  grid[site] = grid[to];
  grid[to] = tmp;
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


using Tuple = std::tuple<int, int, int>;

std::vector<std::pair<Tuple, Tuple>> generate_pairs(int L) {
    std::vector<std::pair<Tuple, Tuple>> pairs;
    std::vector<Tuple> tuples;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                tuples.push_back(std::make_tuple(i, j, k));
            }
        }
    }
    for(int t = 0; t < tuples.size(); ++t)
    {
        const auto [ti, tj, tk] = tuples[t];
        const auto nn = get_neighbors_triplet(ti, tj, tk, L);
        for (const auto & n : nn)
        {
            const auto t1 = tuples[t];
            const auto t2 = std::tuple<int, int, int>(n[0], n[1], n[1]);
            pairs.push_back(std::make_pair(t1, t2));
        }
    }
    return pairs;
}

#define ONETWO(x) ((x & 1) || (x & 2))
static inline float local_energy(int *__restrict grid, const int &N,
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

static inline const float nn_e(int *__restrict lattice, const int N,
                                    const int &i, const int &j, const int &k) {
  const auto site = k + N * (j + N * i);
  float res = local_energy(lattice, N, i, j, k);
  const auto nne = revert_neighbor_table[site];
  for (const auto &[m, l, n] : nne)
    res += local_energy(lattice, N, m, l, n);
  return res;
}




int main() {  
  int L = 20;
  const auto tuples = generate_pairs(L);
  std::ifstream infile("config.txt");
  std::string line;
  getline(infile, line);
  std::stringstream lineStream(line);

  ::nn = get_nn_list(L);
  ::revert_table = generate_revert_table(L);
  ::revert_neighbor_table = generate_neighbor_revert_table(L);



  int num;
  int * grid = (int *)malloc(sizeof(int) * L * L * L);
  int pos = 0;
  while (lineStream >> num)
    grid[pos++] = num;

  std::vector<float> des;
  for (const auto & [site1, site2] : tuples)
  {      
      const auto [si, sj, sk] = site1;
      const auto [mi, mj, mk] = site2;
      const auto site = sk + L * (sj + si * L);
      const auto mv   = mk + L * (mj + mi * L);
      if (grid[site] == grid[mv]) continue;  
      const float E1 = nn_e(grid, L, si, sj, sk) + nn_e(grid,  L, mi, mj, mk);      
      exchange(grid, L, site, mv);
      const float E2 = nn_e(grid, L, si, sj, sk) + nn_e(grid,  L, mi, mj, mk);
      exchange(grid, L, site, mv);
      des.push_back((E2 - E1));
  }
  for(const auto & dE : des)
      std::cout << dE << " ";
  std::cout << "\n"; 
  return 0;
}
