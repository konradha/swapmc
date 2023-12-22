#include <iostream>
#include <omp.h>

void print_slice(const int *__restrict lattice, const int N, const int i) {
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k)
      std::cout << lattice[k + N * (j + i * N)] << " ";
    std::cout << "\n";
  }
}

int main() {
  omp_set_num_threads(4);
  int N = 30;
  int *lat = nullptr;
  const unsigned int align = 64;
  auto padded_N = (N * N * N + (align - 1)) & ~(align - 1);
  if (posix_memalign((void **)&lat, align, padded_N * sizeof(int)) != 0)
    return 1;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        lat[k + N * (j + i * N)] = 0;

  int ii, jj, kk;
  ii = jj = kk = 0;
  for (ii = 0; ii < 3; ++ii)
#pragma omp parallel for collapse(1) schedule(static)
    for (int i = ii; i < N - 3 + 1; i += 3) {
      int m = (i > 2 ? 2 : 0);
      for (int j = jj; j < N - 3 + 1 + m; ++j) {
        int n = (i > 2 ? 2 : 0);
        for (int k = kk; k < N - 3 + 1 + n; ++k) {
          const auto site = k + N * (j + i * N);
          lat[site] = 1;
        }
      }
    }

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        lat[k + N * (j + i * N)] = 0;

#pragma omp parallel for collapse(2)
  for (int ii = 0; ii < N; ii += 3) {
    for (int jj = 0; jj < N; jj += 3) {
      for (int kk = 0; kk < N; kk += 3) {
        for (int i = ii; i < std::min(ii + 3, N); i++) {
          for (int j = std::max(0, jj - ii % 3); j < std::min(jj + 3, N); j++) {
            for (int k = std::max(0, kk - ii % 3); k < std::min(kk + 3, N);
                 k++) {
              auto site = k + N * (j + i * N);
              int l, r, u, d, f, b; // left, right, up, down, front, back
              l = r = u = d = f = b = 0;
              if (i == 0) {
                u = k + N * (j + (N - 1) * N);
              }
              if (i == N - 1) {
                d = k + N * j;
              }
              if (j == 0) {
                l = k + N * ((N - 1) + i * N);
              }
              if (j == N - 1) {
                r = k + N * (i * N);
              }
              if (k == 0) {
                f = N - 1 + N * (j + i * N);
              }
              if (k == N - 1) {
                b = N * (j + i * N);
              }

              if (i > 0) {
                u = k + N * (j + (i - 1) * N);
              }
              if (i < N - 1) {
                d = k + N * (j + (i + 1) * N);
              }
              if (j > 0) {
                l = k + N * (j - 1 + i * N);
              }
              if (j < N - 1) {
                r = k + N * (j + 1 + i * N);
              }
              if (k > 0) {
                f = k - 1 + N * (j + i * N);
              }
              if (k < N - 1) {
                b = k + 1 + N * (j + i * N);
              }

              int nn[6] = {u, d, l, r, f, b};

              for (const auto &n : nn)
                lat[n] += -1;
              //lat[site] = 10;
            }
          }
        }
      }
    }
  }
  int s = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        s += lat[k + N * (j + i * N)];
          
  for (int i = 0; i < N; ++i) {
    print_slice(lat, N, i);
    std::cout << "\n";
  }
  std::cout << "s " << s << "\n";
  std::cout << "o " << -s / (N*N*N) << "\n";
  free(lat);
}
