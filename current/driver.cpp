#include "lattice.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <numeric>

#include <fstream>

#include <random>
#include <stdexcept>
#include <stdlib.h>
#include <string>

#define GET_IDX(nx, ny, nz, i, j, k) (k + ny * (j + i * nx))

struct tru {
  bool operator()(bool b) const { return b; }
};

struct neg {
  bool operator()(bool b) const { return !b; }
};

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

float get_local_energy(Lattice *L, int site) {
  auto E = L->grid[site].energy;
  for (const auto &idx : L->grid[site].neighbors) {
    if (idx == -1)
      continue;
    E += L->grid[idx].energy;
  }
  return E;
}

template <Spin spin> void move(Lattice *L, int from, int to);

template <> void move<Spin::red>(Lattice *L, int from, int to) {
  L->vacant[from] = 1;
  L->red[from] = 0;
  L->grid[from].spin = Spin::empty;
  L->grid[from].energy = 0;

  L->vacant[to] = 0;
  L->red[to] = 1;
  L->grid[to].spin = Spin::red;
  L->grid[to].energy = local_energy(L, to);
}

template <> void move<Spin::blue>(Lattice *L, int from, int to) {
  L->vacant[from] = 1;
  L->blue[from] = 0;
  L->grid[from].spin = Spin::empty;
  L->grid[from].energy = 0;

  L->vacant[to] = 0;
  L->blue[to] = 1;
  L->grid[to].spin = Spin::blue;
  L->grid[to].energy = local_energy(L, to);
}

void move(Lattice *L, int from, int to) {
  switch (L->grid[from].spin) {
  case Spin::red:
    move<Spin::red>(L, from, to);
    break;
  case Spin::blue:
    move<Spin::blue>(L, from, to);
    break;
  case Spin::empty:
    throw std::runtime_error("Attempted to move from an empty slot.");
  }
}

// :: could be varied according to e.g. some radial factor
std::tuple<float, float> flip_and_calc(Lattice *L, int from, int to) {
  // make a move, calculate energy
  move(L, from, to);
  float E2 = get_local_energy(L, to) + get_local_energy(L, from);
  // move back, calculate energy
  move(L, to, from);
  float E1 = get_local_energy(L, from);
  return {E1, E2};
}

std::tuple<int, int> mc_step(Lattice *L) {
  // this now is a local-only move
  std::random_device rd;
  std::mt19937 gen(rd());
  // std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_int_distribution<> dis(0, 5);

  auto mv = sample(L->vacant, L->num_sites, L->num_red + L->num_blue,
                   neg());                   // get one of the particles
  auto to = L->grid[mv].neighbors[dis(gen)]; // get one neighbor
  if (L->grid[to].spin != Spin::empty)
    return {-1, -1}; // cannot move as it's occupied

  auto [E1, E2] = flip_and_calc(L, mv, to);
  auto dE = E2 - E1;
  if (dis(gen) < std::exp(-L->beta * dE)) // make fast
  {
    move(L, mv, to);
    return {mv, to};
  }

  return {-1, -1};
}

void swap(Lattice *L, int r, int b) {
  L->red[r] = 0;
  L->blue[b] = 0;
  L->red[b] = 1;
  L->blue[r] = 1;
  L->grid[r].spin = Spin::blue;
  L->grid[b].spin = Spin::red;

  L->grid[r].energy = L->grid[b].energy = 0;
  L->grid[r].energy = local_energy(L, r);
  L->grid[b].energy = local_energy(L, b);
}

std::tuple<float, float> swap_and_calc(Lattice *L, int r, int b) {
  swap(L, r, b);
  float E2 = get_local_energy(L, r) + get_local_energy(L, b);
  swap(L, b, r);
  float E1 = get_local_energy(L, r);
  return {E1, E2};
}

std::tuple<int, int> swap_step(Lattice *L) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  auto mv = sample(L->red, L->num_sites, L->num_red, tru());   // get a red
  auto to = sample(L->blue, L->num_sites, L->num_blue, tru()); // get a blue
  auto [E1, E2] = swap_and_calc(L, mv, to);
  auto dE = E2 - E1;

  if (dis(gen) < std::exp(-L->beta * dE)) // make fast
  {
    swap(L, mv, to);
    return {mv, to};
  }
  return {-1, -1};
}

float energy(Lattice *L) {
  float E = 0.;
  for (int site = 0; site < L->num_sites; ++site)
    E += L->grid[site].energy;
  return E;
}

int run(float b = 2., float rho = .45, int nstep = 10000,
        float swap_proba = .2) {
  std::random_device rd;
  std::mt19937 gen(__rdtsc());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  float beta = b;
  float n = rho;
  float n_1 = .8 * n;
  int l = 30;
  auto L = Lattice();
  int *pref = (int *)calloc(sizeof(int), 3);
  pref[0] = 0;
  pref[1] = 3;
  pref[2] = 5;
  fill_lattice(&L, beta, n, n_1, l, l, l, pref, 3);

  int nsteps = nstep;

  int from_mc, to_mc;
  int from_swap, to_swap;

  std::vector<std::tuple<int, int, int>> mc_moves;
  std::vector<std::tuple<int, int, int>> swap_moves;
  mc_moves.reserve(nsteps);
  swap_moves.reserve(nsteps);

  for (int i = 0; i < nsteps; ++i) {

    from_mc = to_mc = -1;
    from_swap = to_swap = -1;

    std::tie(from_mc, to_mc) = mc_step(&L);
    mc_moves.push_back({i, from_mc, to_mc});
    if (dis(gen) <= swap_proba)
      std::tie(from_swap, to_swap) = swap_step(&L);
    swap_moves.push_back({i, from_swap, to_swap});
  }
  bool print_it = true;
  finalize(&L, nsteps, pref, mc_moves, swap_moves, print_it);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    run(2., .75, 10, .2);
    std::cout
        << "args (order): temperature, density, #sweeps, swap probability\n";
  } else {
    auto arg_beta = argv[1];
    auto arg_rho = argv[2];
    auto nsteps = argv[3];
    auto swap_prb = argv[4];
    auto beta = atof(arg_beta);
    auto rho = atof(arg_rho);
    auto stps = atof(nsteps);
    auto draw = atof(swap_prb);
    run(beta, rho, stps, draw);
  }
}
