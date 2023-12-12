#include <cassert>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <tuple>
#include <vector>
#define GET_IDX(nx, ny, nz, i, j, k) (k + ny * (j + i * nx))

struct tru {
  bool operator()(bool b) const { return b; }
};

struct neg {
  bool operator()(bool b) const { return !b; }
};


auto _ = -1;
enum class Spin { empty, red, blue };

struct lattice_entry {
  int x;
  int y;
  int z;
  Spin spin;
  int neighbors[6] = {_, _, _, _, _, _};
  float energy;
} __attribute__((packed, aligned(8)));

struct Lattice {
  float beta;
  float n;
  float n1;
  float n2;
  int nx;
  int ny;
  int nz;
  float N;

  lattice_entry *grid;
  // this as SoA
  bool *vacant;
  bool *red;
  bool *blue;

  int num_red;
  int num_blue;
  int num_spins;
  int num_sites;

  int *preferred;
  int pref_size;
};

void print_lattice(lattice_entry *l, int nx, int ny, int nz) {
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        std::cout << l[i + (j + k * ny) * nz].x << " ";
      }
      std::cout << "\n";
    }
    for (int l = 0; l < nz; ++l)
      std::cout << "-";
    std::cout << "\n";
  }
}

void print_sites(Lattice *L) {
  for (int i = 0; i < L->num_sites; ++i) {
    if (L->grid[i].spin == Spin::red)
      std::cout << 2 << " ";
    else if (L->grid[i].spin == Spin::blue)
      std::cout << 1 << " ";
    else
      std::cout << 0 << " ";
  }
  std::cout << "\n";
}

void finalize(Lattice *L, int nsteps, int *pref,
              const std::vector<std::tuple<int, int, int>> &mc_moves,
              const std::vector<std::tuple<int, int, int>> &swap_moves,
              const bool &print_it) {
  int from_mc, to_mc, from_swap, to_swap;
  from_mc = to_mc = from_swap = to_swap = -1;

  if (print_it) {
    print_sites(L);
    int metropolis_epoch = 0;
    int swap_epoch = 0;
    for (int i = 0; i < nsteps; ++i) {
      std::tie(metropolis_epoch, from_mc, to_mc) = mc_moves[i];
      std::tie(swap_epoch, from_swap, to_swap) = swap_moves[i];
      // potential optimization: compress data that's saved
      // if (from_mc == -1 && from_swap == -1) continue;
      std::cout << i << "," << from_mc << "," << to_mc << "," << from_swap
                << "," << to_swap << "\n";
    }
  }
  free(pref);
  free(L->grid);
  free(L->vacant);
  free(L->red);
  free(L->blue);
}

std::string strspin(Spin s) {
  if (s == Spin::red)
    return std::string("red");
  if (s == Spin::blue)
    return std::string("blue");
  return std::string("-");
}

static inline void fill_neighbors(lattice_entry *e, int i, int j, int k, int nx,
                                  int ny, int nz) {
  // wrap around, PBC
  if (i == 0)
    e->neighbors[0] = k + ny * (j + (nx - 1) * nx);
  if (i == nx - 1)
    e->neighbors[3] = k + ny * (j + 0 * nx);
  if (j == 0)
    e->neighbors[1] = k + ny * ((ny - 1) + i * nx);
  if (j == ny - 1)
    e->neighbors[4] = k + ny * (0 + i * nx);
  if (k == 0)
    e->neighbors[2] = (nz - 1) + ny * (j + i * nx);
  if (k == nz - 1)
    e->neighbors[5] = 0 + ny * (j + i * nx);

  // all internal neighbors
  if (i > 0)
    e->neighbors[0] = k + ny * (j + (i - 1) * nx);
  if (j > 0)
    e->neighbors[1] = k + ny * (j - 1 + i * nx);
  if (k > 0)
    e->neighbors[2] = k - 1 + ny * (j + i * nx);
  if (i < nx - 1)
    e->neighbors[3] = k + ny * (j + (i + 1) * nx);
  if (j < ny - 1)
    e->neighbors[4] = k + ny * (j + 1 + i * nx);
  if (k < nz - 1)
    e->neighbors[5] = k + 1 + ny * (j + i * nx);
}

lattice_entry *build_cubic(int nx, int ny, int nz) {
  int N = nx * ny * nz;
  lattice_entry *L = (lattice_entry *)calloc(N, sizeof(lattice_entry));
  // fill connections
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        auto idx = GET_IDX(nx, ny, nz, i, j, k);
        auto site = &L[idx];
        fill_neighbors(site, i, j, k, nx, ny, nz);
        site->x = i;
        site->y = j;
        site->z = k;
        site->energy = 0.;
      }
    }
  }
  return L;
}

std::random_device rd_sample;
std::mt19937 gen_sample(__rdtsc());
template <typename pred> int sample(bool *a, int size, int count, pred p) {
  std::uniform_int_distribution<> dis(0, count - 1);
  int rand = dis(gen_sample);
  for (int i = 0; i < size; ++i) {
    if (p(a[i])) {
      if (rand == 0)
        return i;
      --rand;
    }
  }
  // unreachable
  throw std::runtime_error("Unexpected error in sample function");
}

float local_energy(Lattice *L, int site) {
  if (L->vacant[site])
    return 0.;
  float connection = (float)(L->red[site] ? L->preferred[1] : L->preferred[2]);
  float current_connections = 0.;

  for (const auto &idx : L->grid[site].neighbors) {
    if (idx == -1 || L->vacant[idx])
      continue;
    current_connections += 1.;
  }
  float p = connection - current_connections;
  return p * p;
}



void fill_lattice(Lattice *L, float beta, float n, float N, int nx, int ny,
                  int nz, int *pref, int pref_size) {
  L->beta = beta;
  L->n = n;
  L->n1 = N;
  L->grid = build_cubic(nx, ny, nz);
  L->N = N;
  L->num_sites = nx * ny * nz;
  L->num_spins = (int)(n * L->num_sites);
  L->num_red = (int)(L->n1 * L->num_sites);
  L->num_blue = L->num_spins - L->num_red;

  L->preferred = pref;
  L->pref_size = pref_size;
  L->vacant = (bool *)(calloc(L->num_sites, sizeof(bool)));
  L->red = (bool *)(calloc(L->num_sites, sizeof(bool)));
  L->blue = (bool *)(calloc(L->num_sites, sizeof(bool)));
  for (int i = 0; i < L->num_sites; ++i)
    L->vacant[i] = 1;
  for (int i = 0; i < L->num_sites; ++i)
    L->red[i] = 0;
  for (int i = 0; i < L->num_sites; ++i)
    L->blue[i] = 0;

  int num_reds, num_blues;
  num_reds = num_blues = 0;
  while (num_reds < L->num_red) {
    try {
      auto site = sample(L->vacant, L->num_sites,
                         L->num_sites - (num_reds + num_blues), tru());
      L->vacant[site] = 0;
      L->red[site] = 1;
      L->grid[site].spin = Spin::red;
      num_reds++;
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return;
    }
  }

  while (num_blues < L->num_blue) {
    try {
      auto site = sample(L->vacant, L->num_sites,
                         L->num_sites - (num_reds + num_blues), tru());
      L->vacant[site] = 0;
      L->blue[site] = 1;
      L->grid[site].spin = Spin::blue;
      num_blues++;
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return;
    }
  }

  assert(num_blues == L->num_blue && num_reds == L->num_red);

  for (int i = 0; i < L->num_sites; ++i)
    L->grid[i].energy = local_energy(L, i);
}


