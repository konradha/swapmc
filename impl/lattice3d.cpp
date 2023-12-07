#include "structs.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <immintrin.h>
#include <numeric>


#include <random>
#include <stdexcept>
#include <stdlib.h>
#include <string>

#include <tuple>
#include <vector>


#define GET_IDX(nx, ny, nz, i, j, k) (k + ny * (j + i * nx))


struct tru {
    bool operator()(bool b) const { return b;}
};


struct neg {
    bool operator()(bool b) const { return !b;}
};

template <typename pred>
int sample(bool* a, int size, pred p){
    // TODO this loop is not needed -> we always know
    // L->num_spins, L->num_red, L->num_blue, L->numspins - ...
    int count = 0;
    for (int i = 0; i < size; ++i) {
        if (p(a[i])) {
            ++count;
        }
    }

    if (count == 0) {
        //for(int i=0;i<size;++i) std::cout << a[i] << " ";
        //std::cout << "\n";
        throw std::runtime_error("cannot sample from array");
    }


    // TODO this draw may be done fast(er) ....
    std::random_device rd;
    std::mt19937 gen(__rdtsc());
    std::uniform_int_distribution<> dis(0, count - 1);

    int rand = dis(gen);
    for (int i = 0; i < size; ++i) {
        if (p(a[i])) {
            if (rand == 0) {
                return i;
            }
            --rand;
        }
    }

    // This point should never be reached.
    throw std::runtime_error("Unexpected error in sample function");
}

void print_lattice(lattice_entry *l, int nx, int ny, int nz)
{
    for(int i=0; i<nx;++i) {
        for (int j=0;j<ny;++j){
            for (int k=0;k<nz;++k) {
                std::cout << l[i + (j + k*ny) * nz].x << " ";
            }
            std::cout << "\n";
        }
        for(int l=0;l<nz;++l) std::cout << "-";
        std::cout << "\n";
    }
}

static inline void fill_neighbors(lattice_entry *e, int i, int j, int k,
                    int nx, int ny, int nz)
{
    // wrap around, PBC
    if(i==0)    e->neighbors[0] = k + ny * (j + (nx-1) * nx);
    if(i==nx-1) e->neighbors[3] = k + ny * (j + 0 * nx);
    if(j==0)    e->neighbors[1] = k + ny * ((ny-1) + i * nx);
    if(j==ny-1) e->neighbors[4] = k + ny * (0 + i * nx);
    if(k==0)    e->neighbors[2] = (nz-1) + ny * (j + i * nx);
    if(k==nz-1) e->neighbors[5] = 0 + ny * (j + i * nx);

    // all internal neighbors
    if(i>0)     e->neighbors[0] = k + ny * (j + (i-1) * nx); 
    if(j>0)     e->neighbors[1] = k + ny * (j-1 + i * nx);
    if(k>0)     e->neighbors[2] = k-1 + ny * (j + i * nx);
    if(i<nx-1)  e->neighbors[3] = k + ny * (j + (i+1) * nx);
    if(j<ny-1)  e->neighbors[4] = k + ny * (j+1 + i * nx);
    if(k<nz-1)  e->neighbors[5] = k+1 + ny * (j + i * nx);
}

lattice_entry *build_cubic(int nx, int ny, int nz)
{
    int N = nx * ny * nz;
    lattice_entry *L = (lattice_entry *)calloc(N, sizeof(lattice_entry));
    // fill connections
    for (int i=0;i<nx;++i)
    {
        for (int j=0;j<ny;++j)
        {
            for (int k=0;k<nz;++k)
            { 
                auto idx  = GET_IDX(nx, ny, nz, i, j, k);
                auto site = &L[idx];
                fill_neighbors(site, i, j, k, nx, ny, nz);  
                site->x = i; site->y = j; site->z = k;
                site->energy = 0.;
            }
        }
    }
    return L;
}


float local_energy(Lattice *L, int site)
{
    if (L->vacant[site]) return 0.;    
    float connection = (float)(L->red[site]? L->preferred[1] : L->preferred[2]);
    float current_connections = 0.;
    
    for(const auto &idx : L->grid[site].neighbors)
    {
        if (idx == -1 || L->vacant[idx]) continue;
        current_connections += 1.;
    }
    float p = connection - current_connections;
    return p*p;
}

void fill_lattice(Lattice *L, float beta, float n, float N,
                  int nx, int ny, int nz, int *pref, int pref_size)
{
    std::cout << "total density is: " << n << "\n";
    std::cout << "subdensity is:    " << N << "\n";
    L->beta = beta; L->n = n; L->n1 = N; L->grid = build_cubic(nx, ny, nz);
    L->N = N;  
    L->num_sites = nx * ny * nz;
    L->num_spins = (int)(n*L->num_sites);
    L->num_red = (int)(L->n1 * L->num_sites); L->num_blue = L->num_spins - L->num_red;
    
    L->preferred = pref; L->pref_size = pref_size;    
    L->vacant = (bool*)(calloc(L->num_sites, sizeof(bool)));
    L->red    = (bool*)(calloc(L->num_sites, sizeof(bool)));
    L->blue   = (bool*)(calloc(L->num_sites, sizeof(bool)));
    for (int i=0;i<L->num_sites;++i) L->vacant[i] = 1;
    for (int i=0;i<L->num_sites;++i) L->red   [i] = 0;
    for (int i=0;i<L->num_sites;++i) L->blue  [i] = 0;
  
    int num_reds, num_blues; num_reds = num_blues = 0; 
    while (num_reds < L->num_red)
    {
        try {
            auto site = sample(L->vacant, L->num_sites, tru());
            L->vacant[site] = 0; L->red[site] = 1;
            L->grid[site].spin = Spin::red;
            num_reds++;     
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return;
        } 
    }

    while (num_blues < L->num_blue)
    {
        try {
            auto site = sample(L->vacant, L->num_sites, tru());
            L->vacant[site] = 0; L->blue[site] = 1;
            L->grid[site].spin = Spin::blue;
            num_blues++;     
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return;
        } 
    }
    
    for(int i=0;i<L->num_sites;++i)
        L->grid[i].energy = local_energy(L, i);
}

float get_local_energy(Lattice *L, int site)
{
   auto E  = L->grid[site].energy; 
   for (const auto& idx : L->grid[site].neighbors) {
       if (idx == -1) continue;
       E += L->grid[idx].energy;
   }
   return E;
}

std::string strspin(Spin s)
{
    if (s == Spin::red)  return std::string("red"); 
    if (s == Spin::blue) return std::string("blue");
    return std::string("-");
}

template<Spin spin>
void move(Lattice *L, int from, int to);

template<>
void move<Spin::red>(Lattice *L, int from, int to) {
    L->vacant[from] = 1;
    L->red[from] = 0;
    L->grid[from].spin = Spin::empty;
    L->grid[from].energy = 0;

    L->vacant[to] = 0;
    L->red[to] = 1;
    L->grid[to].spin = Spin::red;
    L->grid[to].energy = local_energy(L, to);
}

template<>
void move<Spin::blue>(Lattice *L, int from, int to) {
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
            //std::cout << strspin(L->grid[from].spin) << "\n";
            //std::cout << L->vacant[from] << "\n";
            //std::cout << L->red[from] << "\n";
            //std::cout << L->blue[from] << "\n";
            throw std::runtime_error("Attempted to move from an empty slot.");
    }
}

// :: could be varied according to e.g. some radial factor
std::tuple<float, float> flip_and_calc(Lattice *L, int from, int to)
{
    // make a move, calculate energy
    move(L, from, to);
    float E2 = get_local_energy(L, to) + get_local_energy(L, from); 
    // move back, calculate energy
    move(L, to, from);
    float E1 = get_local_energy(L, from);
    return {E1, E2};
}


bool mc_step(Lattice *L)
{
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); 
       
    auto mv = sample(L->vacant, L->num_sites, neg()); // get one of the particles
    //std::cout << "FROM;"; 
    //std::cout << " spin: " << strspin(L->grid[mv].spin);
    //std::cout << " vact: " << L->vacant[mv];
    //std::cout << " red:  " << L->red[mv];
    //std::cout << " blue: " << L->blue[mv] << "\n";
    auto to = sample(L->vacant, L->num_sites, tru()); // get one free slot on lattice
    //std::cout << "TO; "; 
    //std::cout << " spin: " << strspin(L->grid[to].spin);
    //std::cout << " vact: " << L->vacant[to];
    //std::cout << " red:  " << L->red[to];
    //std::cout << " blue: " << L->blue[to] << "\n";
    auto [E1, E2] = flip_and_calc(L, mv, to); 
    auto dE = E2 - E1;
    if (dis(gen) < std::exp( -L->beta * dE )) // make fast
    {
        move(L, mv, to);
        return true;
    }

    return false;
}

void swap(Lattice *L, int r, int b)
{ 
    L->red[r] = 0; L->blue[b] = 0;
    L->red[b] = 1; L->blue[r] = 1;
    L->grid[r].spin = Spin::blue;
    L->grid[b].spin = Spin::red;

    L->grid[r].energy = L->grid[b].energy = 0;
    L->grid[r].energy = local_energy(L, r);
    L->grid[b].energy = local_energy(L, b);
}

std::tuple<float, float> swap_and_calc(Lattice *L, int r, int b)
{
    swap(L, r, b);
    float E2 = get_local_energy(L, r) + get_local_energy(L, b);  
    swap(L, b, r);
    float E1 = get_local_energy(L, r);
    return {E1, E2};
}


bool swap_step(Lattice *L)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); 
    auto mv = sample(L->red, L->num_sites,  tru()); // get a red
    auto to = sample(L->blue, L->num_sites, tru()); // get a blue
    auto [E1, E2] = swap_and_calc(L, mv, to); 
    auto dE = E2 - E1;

    if (dis(gen) < std::exp( -L->beta * dE )) // make fast
    {
        swap(L, mv, to);
        return true;
    }
    return false;
}

float energy(Lattice *L)
{
    float E = 0.;
    for (int site = 0; site < L->num_sites; ++site)
        E += L->grid[site].energy;
    return E;
}


void and_arrays(bool * __restrict r, bool * __restrict o, bool * __restrict n, int N)
{
    // still the fastest when running `-O3`
    // -- better than hand-optimized intrinsics
    int k = 0;
    for(;k<N;++k)
        r[k] = o[k] & n[k];
}

float sum_bool(bool * __restrict r, int N)
{
    return static_cast<float>(std::accumulate(r, r+N, 0));
}



float autocorr_simple(
        bool * __restrict tmp1, bool * __restrict tmp2,
        std::vector<bool *> &configs_red, std::vector<bool *> &configs_blue, Lattice *L, int curr, int window_size)
{
    unsigned int align = 32;
    auto padded_N = (L->num_sites + (align-1)) & ~(align-1);
    bool *res_r;bool *res_b;
    if (posix_memalign((void**)&res_r, align, padded_N * sizeof(bool)) != 0) return -100.; // tmp array 
    if (posix_memalign((void**)&res_b, align, padded_N * sizeof(bool)) != 0) return -100.; // tmp array
                                                                                       //
    memcpy(res_r, L->red, L->num_sites);
    memcpy(res_b, L->blue, L->num_sites);

    // TODO better strategy: this tops outat about 2.5e5
    configs_red.push_back(res_r);
    configs_blue.push_back(res_b);

    if (curr < window_size) return -1.;

    auto N  = L->num_sites * L->n;
    auto C0 = L->n * (L->n1 * L->n1 + L->n2 * L->n2);

    and_arrays(tmp1, res_r, configs_red[curr-window_size], L->num_sites);    
    and_arrays(tmp2, res_b, configs_blue[curr-window_size], L->num_sites);
    
    auto r = sum_bool(tmp1, L->num_sites);
    auto b = sum_bool(tmp2, L->num_sites);

    auto div = (1./N) * (r + b) - C0;
    return div / (1. - C0);
}

int run(float b=2., float rho=.45, int nstep=10000)
{
    std::random_device rd;
    std::mt19937 gen(__rdtsc());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    float beta = b; float n = rho; float n_1 = .8 * n;
    int l = 30; 
    auto L = Lattice();
    int *pref =(int*)calloc(sizeof(int), 3);
    pref[0] = 0; pref[1] = 3; pref[2] = 5;
    fill_lattice(&L, beta, n, n_1, l, l, l, pref, 3); 


    unsigned int align = 32;
    auto padded_N = (L.num_sites + (align-1)) & ~(align-1);
    
    int autocorr_window = 128;    
    bool *tmp1; bool *tmp2; 
    if (posix_memalign((void**)&tmp1, align, padded_N * sizeof(bool)) != 0) return 1; // tmp array 
    if (posix_memalign((void**)&tmp2, align, padded_N * sizeof(bool)) != 0) return 1; // tmp array
    std::vector<bool *> configs_red; 
    std::vector<bool *> configs_blue;
    
    int nsteps = nstep;
       
    
    for (int i=0; i<nsteps;++i)
    {
        auto ac = autocorr_simple(tmp1, tmp2, configs_red, configs_blue, &L, i, autocorr_window); 
        auto e = energy(&L); auto epp = e / L.num_spins;
        std::cout << i << "," << e << "," <<epp<< ","<<ac << "\n";
        mc_step(&L);
        if (dis(gen) <= .12) swap_step(&L); 
    }

    for (int i=0;i<configs_red.size();++i) free(configs_red[i]);
    for (int i=0;i<configs_blue.size();++i) free(configs_blue[i]);
    free(pref);
    free(tmp1); free(tmp2); 
    free(L.grid);
    free(L.vacant); free(L.red); free(L.blue);
    return 0;
}
                                      

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        run(2.,.75,10);
        std::cout << "args (order): temperature, density, #sweeps\n";
    }
    else
    {
        auto arg_beta = argv[1];
        auto arg_rho  = argv[2];
        auto nsteps   = argv[3];
        auto beta = atof(arg_beta);
        auto rho  = atof(arg_rho);
        auto stps = atof(nsteps);
        run(beta, rho, stps);
    }
}
