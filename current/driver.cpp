#include "lattice.h"
#include <cmath>
#include <stdexcept>


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
  float E2 = local_energy(L, to) + local_energy(L, from);
  // move back, calculate energy
  move(L, to, from);
  float E1 = local_energy(L, from);
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
  float E2 = local_energy(L, r) + local_energy(L, b);
  swap(L, b, r);
  float E1 = local_energy(L, r);
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

void dump_data(std::vector<std::tuple<int, int>> &metro_moves,
               std::vector<std::tuple<int, int>> &swap_moves,
               int n)
{
    int  from_mc, to_mc, from_swap, to_swap;
    from_mc = to_mc = from_swap =  to_swap = -1;

    for (int i = 0; i < n; ++i) {
      std::tie(from_mc, to_mc) = metro_moves[i];
      std::tie(from_swap, to_swap) = swap_moves[i];
      std::cout << i << "," << from_mc << "," << to_mc << "," << from_swap
                << "," << to_swap << "\n";
    }
    metro_moves.clear(); swap_moves.clear();
    metro_moves.reserve(n); swap_moves.reserve(n);
}

int run(float b = 2., float rho = .45, int nstep = 10000, int xstep=10000,
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

  size_t rsv = 1000000;
  std::vector<std::tuple<int, int>> mc_moves;
  std::vector<std::tuple<int, int>> swap_moves;
  mc_moves.reserve(rsv);
  swap_moves.reserve(rsv);

  // THERMALIZE
  for( int i=0;i<xstep;++i) mc_step(&L); 

  print_sites(&L);
  // SAMPLE LOOP
  for (int i = 0; i < nsteps; ++i) {

    from_mc = to_mc = -1;
    from_swap = to_swap = -1;

    std::tie(from_mc, to_mc) = mc_step(&L);
    mc_moves.push_back({from_mc, to_mc});
    if (dis(gen) <= swap_proba)
      std::tie(from_swap, to_swap) = swap_step(&L);
    swap_moves.push_back({from_swap, to_swap});
    if (i > 0 && i % rsv == 0)
        dump_data(mc_moves, swap_moves, rsv);

  }
  bool print_it = true;
  finalize(&L, nsteps, pref, mc_moves, swap_moves, print_it);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    run(2., .75, 10, 10, .2);
    std::cout
        << "args (order): temperature, density, #sweeps, #thermalization sweeps, swap probability\n";
  } else {
    auto arg_beta = argv[1];
    auto arg_rho = argv[2];
    auto nsteps = argv[3];
    auto xsteps = argv[4];
    auto swap_prb = argv[5];
    auto beta = atof(arg_beta);
    auto rho = atof(arg_rho);
    auto stps = atof(nsteps);
    auto xtps = atof(xsteps);
    auto draw = atof(swap_prb);
    run(beta, rho, stps, xtps, draw);
  }
}
