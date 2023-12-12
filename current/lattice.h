#include <iostream>
#include <stdlib.h>
#include <tuple>
#include <vector>

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
              const std::vector<std::tuple<int, int, int>> &swap_moves) {
  int from_mc, to_mc, from_swap, to_swap;
  from_mc = to_mc = from_swap = to_swap = -1;
  print_sites(L);
  int metropolis_epoch = 0;
  int swap_epoch = 0;
  for (int i = 0; i < nsteps; ++i) {
    std::tie(metropolis_epoch, from_mc, to_mc) = mc_moves[i];
    std::tie(swap_epoch, from_swap, to_swap) = swap_moves[i];
    if (from_mc == -1 && from_swap == -1) continue;  
    std::cout << i << "," << from_mc << "," << to_mc << "," << from_swap << ","
              << to_swap << "\n";
     
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
