// clang-15 -lm -lstdc++ -std=c++17 lattice3d.cpp -o lat && ./lat

#include <cmath>
#include <iostream>
#include <vector>

#include <random>
#include <stdlib.h>
#include <tuple>


struct tru {
    bool operator()(bool b) const { return !b;}
};


struct neg {
    bool operator()(bool b) const { return b;}
};

template <typename pred>
int sample(bool* a, int size, pred p) {
    // std::cout << "array is\n";
    // for(auto i=0;i<size;++i) std::cout << a[i] << " ";
    // std::cout << "\n";

    int count = 0;
    for (int i = 0; i < size; ++i) {
        if (p(a[i])) {
            ++count;
        }
    }

    if (count == 0) {
        throw std::runtime_error("cannot sample from array");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
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




auto _ = -1;
enum class Spin {empty, red, blue};
struct lattice_entry {
    int x; int y; int z;
    Spin spin;
    int neighbors[6]={_, _, _, _, _, _};
    float energy;
    //bool boundary_neighbors[6]){_,_,_,_,_,_}; // for parallelization purposes
} __attribute__((packed, aligned(8)));
// TODO benchmark alignment
// TODO check for better granularity in SoA

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

// TODO macro
static inline int get_idx(int nx, int ny, int nz, 
                      int  i, int  j, int  k)
{
    return k + ny * (j + i * nx);
}

// TODO macro
// TODO check if correct
// yields index in lattice_site array to be able to immediately access energy
static inline void fill_neighbors(lattice_entry *e, int i, int j, int k,
                    int nx, int ny, int nz)
{
    if(i>0)    e->neighbors[0] = k + ny * (j + (i-1) * nx); 
    if(j>0)    e->neighbors[1] = k + ny * (j-1 + i * nx);
    if(k>0)    e->neighbors[2] = k-1 + ny * (j + i * nx);
    if(i<nx-1) e->neighbors[3] = k + ny * (j + (i+1) * nx);
    if(j<ny-1) e->neighbors[4] = k + ny * (j+1 + i * nx);
    if(k<nz-1) e->neighbors[5] = k+1 + ny * (j + i * nx);
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
                auto idx  = get_idx(nx, ny, nz, i, j, k);
                auto site = &L[idx];
                fill_neighbors(site, i, j, k, nx, ny, nz);  
                site->x = i; site->y = j; site->z = k;
                site->energy = 0.;
            }
        }
    }
    return L;
}

struct Lattice {
    float beta;
    float n;
    float n1;
    float n2;
    int nx; int ny; int nz;
    int N;

    lattice_entry *grid;
    // this as SoA
    bool *vacant;
    bool *red;
    bool *blue;

    int num_red;
    int num_blue; 
    int num_spins;

    int *preferred;
};


float local_energy(Lattice *L, int site)
{
    if (L->vacant[site]) return 0.;    

    float connection = (float)(L->red[site]? L->preferred[1] : L->preferred[2]);
    float current_connections = 0.;
    for(const auto &idx : L->grid[site].neighbors)
    {
        if (idx == -1 || L->vacant[idx]) continue;
        current_connections += 1.; // TODO should this depend on red/blue?
    }
    float p = connection - current_connections;
    return p*p;
}

void fill_lattice(Lattice *L, float beta, float n, float N,
                  int nx, int ny, int nz)
{
    L->beta = beta; L->n = n; L->n1 = N; L->grid = build_cubic(nx, ny, nz);
    L->N = (int)(N); L->num_spins = (int)(n*N); 
    // TODO check how to distribute particles
    L->num_red = (int)(n*N/2); L->num_blue = L->num_spins - L->num_red;

    L->vacant = (bool*)(calloc(L->num_spins, sizeof(bool)));
    L->red    = (bool*)(calloc(L->num_spins, sizeof(bool)));
    L->blue   = (bool*)(calloc(L->num_spins, sizeof(bool)));
    for (int i=0;i<L->num_spins;++i) L->vacant[i] = 1;
    for (int i=0;i<L->num_spins;++i) L->red   [i] = 0;
    for (int i=0;i<L->num_spins;++i) L->blue  [i] = 0;

    int num_reds, num_blues; num_reds = num_blues = 0;
    // TODO: simulation breaks down here. figure out tomorrow
    while (num_reds < L->num_red)
    {
        try {
            auto site = sample(L->vacant, L->num_spins, tru());
            L->vacant[site] = 0; L->red[site] = 1;
            L->grid[site].spin = Spin::red;
            num_reds++;     
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        } 
    }

    while (num_blues < L->num_blue)
    {
        try {
            auto site = sample(L->vacant, L->num_spins, tru());
            L->vacant[site] = 0; L->blue[site] = 1;
            L->grid[site].spin = Spin::blue;
            num_blues++;     
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        } 
    }
    for(int i=0;i<L->num_spins;++i)
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


////template<> -- for red/blue
void move(Lattice *L, int from, int to)
{
    if (L->red[from]) {
        // auto site_from = L->grid[from];
        // auto site_to   = L->grid[to];
        L->vacant[from] = 1;
        L->vacant[to] = 0;
        L->grid[from].spin = Spin::empty;
        L->grid[from].energy = 0;
        L->vacant[from] = 1;
        L->red[from] = 0;
        L->vacant[to] = 0;
        L->red[to] = 1;
        L->grid[to].spin = Spin::red;
        L->grid[to].energy = local_energy(L, to);
    } else {
        L->vacant[from] = 1;
        L->vacant[to] = 0;
        L->grid[from].spin = Spin::empty;
        L->grid[from].energy = 0;
        L->vacant[from] = 1;
        L->red[from] = 0;
        L->vacant[to] = 0;
        L->red[to] = 1;
        L->grid[to].spin = Spin::red;
        L->grid[to].energy = local_energy(L, to);
    }
}

std::tuple<float, float> flip_and_calc(Lattice *L, int from, int to)
{
    // change to local only?
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
    
    auto mv = sample(L->vacant, L->N, tru()); // get one of the particles
    auto to = sample(L->vacant, L->N, neg()); // get one free slot on lattice
                                              //
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

    L->grid[r].energy = L->grid[r].energy = 0;
    L->grid[r].energy = local_energy(L, r);
    L->grid[b].energy = local_energy(L, b);
}

std::tuple<float, float> swap_and_calc(Lattice *L, int r, int b)
{
    // TODO check if terms missing
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

    auto mv = sample(L->red, L->N,  tru()); // get a red
    auto to = sample(L->blue, L->N, tru()); // get a blue

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
    for (int site = 0; site < L->num_spins; ++site)
        E += L->grid[site].energy;
    return E;
}

void run()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    float beta = 2.; float n = .45; float n_1 = .8 * n;
    int l = 6; 
    auto L = Lattice();
    fill_lattice(&L, beta, n, n_1, l, l, l);
    int nsteps = 2500;
    std::vector<float> energies;
    for (int i=0; i<nsteps;++i)
    {
        mc_step(&L);
        //if (dis(gen) <= .3) swap_step(&L); 
        //energies.push_back(energy(&L));
    }
}



/* TODO
void clean(Lattice *L) 
{
    free(L->
}
*/

                                      

int main()
{
    run();
    //int nx = 4; int ny = 4; int nz = 4;
    //int N = nx * ny * nz;
    //auto L = build_cubic(nx, ny, nz);
    //free(L);
}









// struct lattice_entry {
//     int x; int y; int z;
//     Spin spin;
//     int neighbors[6]={_, _, _, _, _, _};
//     float energy;
//     //bool boundary_neighbors[6]){_,_,_,_,_,_}; // for parallelization purposes
// } __attribute__((packed, aligned(8)));
// struct Lattice {
//     float beta;
//     float n;
//     float n1;
//     float n2;
//     int nx; int ny; int nz;
//     int N;
// 
//     lattice_entry *grid;
//     // this as SoA
//     bool *vacant;
//     bool *red;
//     bool *blue;
// 
//     int num_red;
//     int num_blue; 
//     int num_spins;
// 
//     int *preferred;
// };
