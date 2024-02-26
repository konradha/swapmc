#include "packed_maps.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <cstring>

void build_lattice_diag(const int & num_particles, const int & num_red, const int & num_blue,
                        std::mt19937 & generator, std::uniform_int_distribution<> & indices)
{
    for(int i=0; i<L * L * L; ++i)
        set_value_lattice(i, 0);
    int curr_red, curr_blue;
    curr_red = curr_blue = 0;
    while (curr_red < num_red)
    {        
        const int site = indices(generator); 
        const auto i = revert_table[3 * site + 0];
        const auto j = revert_table[3 * site + 1];
        const auto k = revert_table[3 * site + 2]; 
        if (((i+j+k) & 1) == 0 && static_cast<int>(get_value_lattice(site)) == 0)
        { 
            set_value_lattice(site, 1);
            curr_red++;
            continue;
        } 
    }
    while (curr_blue < num_blue)
    {
        const int site = indices(generator);
        const auto i = revert_table[3 * site + 0];
        const auto j = revert_table[3 * site + 1];
        const auto k = revert_table[3 * site + 2];
        if (((i+j+k) & 1) == 1 && static_cast<int>(get_value_lattice(site)) == 0)
        {
            set_value_lattice(site, 2);
            curr_blue++;
        }
    }
    int nred, nblue; nred = nblue = 0;
    for (int i = 0; i < L * L * L; ++i) {
      if (get_value_lattice(i) == 1)
        nred++;
      else if (get_value_lattice(i) == 2)
        nblue++;
    }      
    assert(nred == num_red);
    assert(nblue == num_blue);
}

void exchange(const int &site,
                            const int &to) {
  const auto tmp = get_value_lattice(site);
  set_value_lattice(site, get_value_lattice(to));
  set_value_lattice(to, tmp);
}

#define ONETWO(x) ((x & 0x1) || (x & 0x2))
static inline float local_energy(const int &i, const int &j, const int &k) {
  const auto site = forward_table[i][j][k];
  if (get_value_lattice(site) == 0)
    return 0.;
  const float connection = get_value_lattice(site) == 1 ? 3. : 5.;
  const int * nn_ptr = &(nearest_neighbors[6 * site]);
  int e = ONETWO(get_value_lattice(nn_ptr[0])) + ONETWO(get_value_lattice(nn_ptr[1])) + ONETWO(get_value_lattice(nn_ptr[2])) 
      + ONETWO(get_value_lattice(nn_ptr[3])) + ONETWO(get_value_lattice(nn_ptr[4])) + ONETWO(get_value_lattice(nn_ptr[5]));
  float current = static_cast<float>(e) - connection;
  return current * current;
  //local_energies[site] = current * current;
  //return local_energies[site];
}

static inline const float nn_energy(const int &i, const int &j, const int &k) {
  const auto site = forward_table[i][j][k];  
  float res = local_energy(i, j, k);
  const int * nne = &(revert_nn_table[6 * 3 * site]);
  for(int i = 0; i < 6; ++i)
  {
      int m = nne[3 * i + 0];
      int l = nne[3 * i + 1];
      int n = nne[3 * i + 2];  
      res += local_energy(m, l, n); 
  }
  return res;
}

static inline float local_energy_nochange(const int &i, const int &j, const int &k, const int &w, const int &idx) {
  const auto site = forward_table[i][j][k];
  if (get_value_lattice(site) == 0)
    return 0.;
  const float connection = get_value_lattice(site) == 1 ? 3. : 5.;
  const int * nn_ptr = &(nearest_neighbors[6 * site]);
  int e = ONETWO(get_value_lattice(nn_ptr[0])) + ONETWO(get_value_lattice(nn_ptr[1])) + ONETWO(get_value_lattice(nn_ptr[2])) 
      + ONETWO(get_value_lattice(nn_ptr[3])) + ONETWO(get_value_lattice(nn_ptr[4])) + ONETWO(get_value_lattice(nn_ptr[5]));
  float current = static_cast<float>(e) - connection;
  current = current * current;
  local_e_update_site[w][idx] = site;
  local_e_update_ey[w][idx] = current;
  return current;
}


const float nn_energy_trial(const int &i, const int &j, const int &k, const int& w)
{
  const auto site = forward_table[i][j][k];  
  float res = local_energy_nochange(i, j, k, w, 0);
  const int * nne = &(revert_nn_table[6 * 3 * site]);
  for(int i = 0; i < 6; ++i)
  {
      int m = nne[3 * i + 0];
      int l = nne[3 * i + 1];
      int n = nne[3 * i + 2];  
      res += local_energy_nochange(m, l, n, w, i + 1); 
  }
  return res;
}

void update_local_energy_table()
{
    for(int i=0;i<2;++i)
    {
        for(int p=0;p<7;++p)
        {
            const auto site  = local_e_update_site[i][p];
            const auto new_e = local_e_update_ey[i][p];
            local_energies[site] = new_e;
        }
    }
}

void nonlocal_sweep(const float &beta, std::mt19937 & generator, std::uniform_int_distribution<> & indices, std::uniform_real_distribution<> &uni)
{
    for(int i=0;i<L*L*L;++i)
    {
      const int site = indices(generator); 
      const auto mv = indices(generator); 
      if (get_value_lattice(site) == get_value_lattice(mv)) continue;
      const auto si = revert_table[3 * site + 0];
      const auto sj = revert_table[3 * site + 1];
      const auto sk = revert_table[3 * site + 2];
      const auto mi = revert_table[3 * mv + 0];
      const auto mj = revert_table[3 * mv + 1];
      const auto mk = revert_table[3 * mv + 2]; 
      const float E1 = nn_energy(si, sj, sk) + nn_energy(mi, mj, mk); 
      exchange(site, mv);
      // apparently not faster :(
      //const float E2 = nn_energy_trial(si, sj, sk, 0) + nn_energy_trial(mi, mj, mk, 1); 
      const float E2 = nn_energy(si, sj, sk) + nn_energy(mi, mj, mk); 
      const float dE = E2 - E1; 
      if (dE <= 0 || uni(generator) < std::exp(-beta * dE)) 
      {
          update_local_energy_table();
          continue;
      }
      exchange(site, mv);
    }
}

int energy()
{
    int e = 0;
    for(int i = 0; i < L;++i)
        for(int j=0;j<L;++j)
            for(int k=0;k<L;++k)
               e += local_energy(i,j,k);
    return e;
}

bool check_cpy(uint8_t * lat)
{
    
    for(int i=0;i<L*L*L/4;++i)
        if(packed_lattice[i] != lat[i]) return false;
    return true;
}


int main()
{
    const float rho = .75;
    // convenience defs
    const float rho1 = .45;
    //const float rho2 = .3;
    const int N = (int)(lat_size * rho);
    const int N1 = (int)(rho1 * lat_size);
    const int N2 = N - N1;
    const auto max_idx = L * L * L - 1;
    const float beta = 6.;
    auto uni = std::uniform_real_distribution<>(0., 1.);
    auto indices = std::uniform_int_distribution<>(0, max_idx);
    auto generator = std::mt19937(); 
    generator.seed(__rdtsc());
    generate_tables();


    build_lattice_diag(N, N1, N2, generator, indices);
    //for(int i=0;i<L*L*L;++i)
    //    std::cout << (int)get_value_lattice(i) << " ";
    //std::cout << "\n";

    uint8_t cpy_lat[L*L*L/4];
    memcpy(cpy_lat, packed_lattice, L*L*L/4); 
    //std::cout << check_cpy(cpy_lat) << ", " << energy() << "\n";
    for(int i=0;i<L*L*L; ++i)
    {
        const auto ii = revert_table[3 * i + 0]; 
        const auto jj = revert_table[3 * i + 1];
        const auto kk = revert_table[3 * i + 2];
        local_energy(ii, jj, kk);
    }


   
    int printer = 1;
    for(int i=0;i<1000;++i)
    {
        nonlocal_sweep(beta, generator, indices, uni);
        if(printer == i)
        {
            //std::cout << check_cpy(cpy_lat) << ", " << energy() << "\n";
            printer *= 2;
        }
    }

    for(int i=0;i<L*L*L;++i)
        std::cout << static_cast<int>(get_value_lattice(i)) << " "; 
    std::cout << "\n";



    return 0;
}
