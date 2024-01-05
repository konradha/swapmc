#include <mpi.h>

#include <iostream>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int numranks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int N = 30;
  int dims[2] = {0, 0};
  MPI_Dims_create(numranks, 2, dims);

  int periods[2] = {1, 1};
  int reorder = 1;

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

  MPI_Finalize();
}
