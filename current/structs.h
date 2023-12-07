auto _ = -1;
enum class Spin {empty, red, blue};

struct lattice_entry {
    int x; int y; int z;
    Spin spin;
    int neighbors[6]={_, _, _, _, _, _};
    float energy;
} __attribute__((packed, aligned(8)));


struct Lattice {
    float beta;
    float n;
    float n1;
    float n2;
    int nx; int ny; int nz;
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

