#include <iostream>
#include <vector>

#include <random>
#include <stdlib.h>

int sample(bool* a, int size) {
    int count = 0;
    for (int i = 0; i < size; ++i) {
        if (!a[i]) {
            ++count;
        }
    }

    if (count == 0) {
        throw std::runtime_error("No true elements in the array");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, count - 1);

    int rand = dis(gen);
    for (int i = 0; i < size; ++i) {
        if (!a[i]) {
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
    int neighbors[6]={_,_,_,_,_,_};
    float energy;
    //bool boundary_neighbors[6]){_,_,_,_,_,_}; // for parallelization purposes
} __attribute__((packed, aligned(8)));
// TODO benchmark alignment

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

    lattice_entry *grid;
    // this could be made better
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

    while (num_reds < L->num_red)
    {
        try {
            auto site = sample(L->vacant, L->num_spins);
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
            auto site = sample(L->vacant, L->num_spins);
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


bool mc_step(Lattice *L)
{
    // TODO
    // - can we actually just copy a single site and then 
    //   compute different energies to accept/reject?
    return false;
}


bool swap_step(Lattice *L)
{
    // TODO
    // - can we actually just copy a single site and then 
    //   compute different energies to accept/reject?
    return false;
}



/* TODO
void clean(Lattice *L) 
{
    free(L->
}
*/

                                      

int main()
{
    int nx = 4; int ny = 4; int nz = 4;
    int N = nx * ny * nz;
    auto L = build_cubic(nx, ny, nz);
    free(L);
}
