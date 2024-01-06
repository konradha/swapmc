#include <mpi.h>
#include <iostream>

// TO RUN WITH __8__ ranks
//
// cube is
//
// top slice
// A B
// C D
// bottom slice
// E F
// G H

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int d = 3;
  int N = 30;
  int dims[d] = {0,0,0};
  int err = MPI_Dims_create(size, d, dims);
  if (err != MPI_SUCCESS) {
    std::cout << "Error in MPI_Dims_create" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  int periods[d] = {1, 1, 1};
  int reorder = 1;

  MPI_Comm cart_comm;
  err = MPI_Cart_create(MPI_COMM_WORLD, d, dims, periods, reorder, &cart_comm);
  if (err != MPI_SUCCESS) {
    std::cout << "Error in MPI_Cart_create" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  int coords[d];
  MPI_Cart_coords(cart_comm, rank, d, coords);

  int up, down, left, right, front, back;
  MPI_Cart_shift(cart_comm, 0, 1, &down, &up);
  MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
  MPI_Cart_shift(cart_comm, 2, 1, &back, &front);

  unsigned char p = 'A';
  for (int i=0;i<size;++i){
    if (rank == i)
    {
      std::cout << p << ", coordinates: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")" << std::endl;
      std::cout << p << " has neighbors (u,d,l,r,b,f): " << up << "," << down << "," << left << "," 
                << right << "," << back << "," << front << "\n";
    }
    p++;
  }

  // first round:  (0, 0, 0); (0, 1, 1); (1, 1, 0); (1, 0, 1)
  // second round: (0, 1, 0); (0, 0, 1); (1, 0, 0); (1, 1, 1)


  


  MPI_Finalize();
}
