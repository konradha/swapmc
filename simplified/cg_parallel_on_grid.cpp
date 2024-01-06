#include <mpi.h>
#include <iostream>
#include <tuple>


/*
 *
 * ACTUAL HALO EXCHANGE:
 * only need to exchange an array of size local_N - 1 per run!
 * then chose a tie-breaker for the middle part (or just introduce a bias ...)
 *
 * DON'T need to exchange x,y,z slices for this to work (only after final run (ie, second)!
 */





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
  if (size != 4) 
      MPI_Abort(MPI_COMM_WORLD, 4);

  const int d = 3;
  int N = 30;
  int local_N = 1 + N / 2 + 1;

  int * lattice = nullptr;
  int * local_lattice1 = nullptr;
  int * local_lattice2 = nullptr;

  const unsigned int align = 64;
  const auto padded_N = (N * N * N + (align - 1)) & ~(align - 1);

  const auto local_padding = (local_N * local_N * local_N + (align - 1)) & ~(align - 1);


  if (posix_memalign((void **)&local_lattice1, align, local_padding * sizeof(int)) != 0)
      return 1;
  if (posix_memalign((void **)&local_lattice2, align, local_padding * sizeof(int)) != 0)
      return 1;


  if (rank == 0)
  {
    if (posix_memalign((void **)&lattice, align, padded_N * sizeof(int)) != 0)
      return 1;
    for (int i=0;i<N;++i)
      for(int j=0;j<N;++j)
        for(int k=0;k<N;++k)
          lattice[k + N * (j + i*N)] = 0;
    
    for (int i=0; i<local_N; ++i)
        for(int j=0;j<local_N; ++j)
            for(int k=0;k<local_N; ++k)
                local_lattice1[k + local_N * (j + i*local_N)] = 0; 
    for (int i=0; i<local_N; ++i)
        for(int j=0;j<local_N; ++j)
            for(int k=0;k<local_N; ++k)
                local_lattice2[k + local_N * (j + i*local_N)] = 0;
  }

  int local_idx_xi_1 = rank == 0 || rank == 1? 1 : N/2+1;
  int local_idx_xi_2 = rank == 0 || rank == 1? N/2+1 : 1;
  int local_idx_yi_1 = rank == 0 || rank == 2? 1 : N/2+1;
  int local_idx_yi_2 = rank == 0 || rank == 2? N/2+1 : 1;
  int local_idx_zi_1 = rank == 0 || rank == 3? 1 : N/2+1;
  int local_idx_zi_2 = rank == 0 || rank == 3? N/2+1 : 1;

  int local_idx_xf_1 = local_idx_xi_1 + N/2 + 1;
  int local_idx_xf_2 = local_idx_xf_1 + N/2 + 1;
  int local_idx_yf_1 = local_idx_yi_1 + N/2 + 1;
  int local_idx_yf_2 = local_idx_yf_1 + N/2 + 1;
  int local_idx_zf_1 = local_idx_zi_1 + N/2 + 1;
  int local_idx_zf_2 = local_idx_zf_1 + N/2 + 1;

   

  std::cout << "rank " << rank << " has intervals "
    << local_idx_xi_1 << "," 
    << local_idx_xf_1 << ","
    << local_idx_yi_1 << ","
    << local_idx_yf_1 << ","
    << local_idx_zi_1 << ","
    << local_idx_zf_1 << "\n";



  // exchanging arrays between run 1 (white) and run 2 (black) -- ie. in between completing the checkerboard

  // 0 and 1 -- 0 exchanges in z dir, fixed final_x, final_y; 1 with start_x, start_y
  // 2 and 3 -- 2 exchanges in z dir, fixed final_x, final_y; 3 with start_x, start_y


  // exchange 1: 
  

  
  int * midbuf, * recv_buf;
  if (posix_memalign((void **)&midbuf, align, local_N/2 * sizeof(int)) != 0)
      return 1;
  if (posix_memalign((void **)&recv_buf, align, local_N/2 * sizeof(int)) != 0)
      return 1;


  // determine indices
  // determine neighbors
  // send appropriately
  // recv appropriately
  // work appropriately into local_lattice1, local_lattice2
  // start round 2 of checkerboard sweep
  // TODO: also needs to account for PBC -- ie. needs to exchange (0, 0, k) with (1, 1, k)
  // ABORT THIS HERE THEN.
  
  int corner_x, corner_y;
  int nb;
  if (rank == 0) {
      corner_x = local_N/2; 
      corner_y = local_N/2;
      nb = 1;
  } else if (rank == 1) {
      corner_x = 0; 
      corner_y = 0;
      nb = 0;
  } else if (rank == 2) {
      corner_x = 0; 
      corner_y = local_N/2;
      nb = 3;
  } else if (rank == 3) {
      corner_x = local_N/2; 
      corner_y = 0;
      nb = 2;
  } else {
      MPI_Abort(MPI_COMM_WORLD, 4);
  }

  for(int i=corner_x;;)
      for(int j=corner_y;;)
          for(int k=0;k<local_N/2;++k)
              midbuf[k] = local_lattice1[k + local_N * (j + i * local_N)];
    
  std::cout << "rank " << rank << " will send/recv with " << nb << "\n";
  //MPI_Request small_reqs[2];
  //MPI_Status  small_stats[2];

  //MPI_Isend(midbuf, local_N, MPI_INT, nb, 0, MPI_COMM_WORLD, &small_reqs[0]); 
  //MPI_Irecv(recv_buf, local_N, MPI_INT, nb, 0, MPI_COMM_WORLD, &small_reqs[1]);
  //MPI_Waitall(2 , small_reqs, small_stats);
 
  














  /*



  // introducing some redundancy here to have consistency where in intervals the dimensions are

  // x-neighbors: left, right -- vary y, z
  const int lhalo_start[3] = {local_idx_xi_1 - 1, (local_idx_yi_1 - 1), (local_idx_zi_1 - 1)};  
  const int lhalo_end  [3] = {local_idx_xi_1 - 1, (local_idx_yf_1 + 1), (local_idx_zf_1 + 1)}; 
  const int rhalo_start[3] = {local_idx_xf_1 + 1, (local_idx_yi_1 - 1), (local_idx_zi_1 - 1)};
  const int rhalo_end  [3] = {local_idx_xf_1 + 1, (local_idx_yf_1 + 1), (local_idx_zf_1 + 1)};

  // y-neighbors: up, down -- vary x, z
  const int uhalo_start[3] = {local_idx_xi_1 - 1, (local_idx_yi_1 - 1), (local_idx_zi_1 - 1)};
  const int uhalo_end  [3] = {local_idx_xf_1 + 1, (local_idx_yi_1 - 1), (local_idx_zf_1 + 1)};
  const int dhalo_start[3] = {local_idx_xi_1 - 1, (local_idx_yf_1 + 1), (local_idx_zi_1 - 1)};
  const int dhalo_end  [3] = {local_idx_xf_1 + 1, (local_idx_yf_1 + 1), (local_idx_zf_1 + 1)};

  // z-neighbors: front, back -- vary x, y
  const int fhalo_start[3] = {local_idx_xi_1 - 1, (local_idx_yi_1 - 1), (local_idx_zi_1 - 1)};
  const int fhalo_end  [3] = {local_idx_xf_1 + 1, (local_idx_yf_1 + 1), (local_idx_zi_1 - 1)};
  const int bhalo_start[3] = {local_idx_xi_1 - 1, (local_idx_yi_1 - 1), (local_idx_zf_1 + 1)};
  const int bhalo_end  [3] = {local_idx_xf_1 + 1, (local_idx_yf_1 + 1), (local_idx_zf_1 + 1)};

  // memcpy's to just let mpi sendrecv handle it contiguously
  // --> need to have local copies for the block of the second round then


  int *lbuf, *rbuf;
  int *ubuf, *dbuf;
  int *fbuf, *bbuf;

  for (auto buf : {lbuf, rbuf, ubuf, dbuf, fbuf, bbuf})
    if (posix_memalign((void **)&buf, align, local_N * local_N * sizeof(int)) != 0)
      return 1;

  int *lbuf_recv, *rbuf_recv;
  int *ubuf_recv, *dbuf_recv;
  int *fbuf_recv, *bbuf_recv;

  for (auto buf : {lbuf_recv, rbuf_recv, ubuf_recv, dbuf_recv, fbuf_recv, bbuf_recv})
    if (posix_memalign((void **)&buf, align, local_N * local_N * sizeof(int)) != 0)
      return 1;



  const auto intervals = {
      std::tuple<const int *, const int *, int *>(lhalo_start, lhalo_end, lbuf),
      std::tuple<const int *, const int *, int *>(rhalo_start, rhalo_end, rbuf),

      std::tuple<const int *, const int *, int *>(uhalo_start, uhalo_end, ubuf),
      std::tuple<const int *, const int *, int *>(dhalo_start, dhalo_end, dbuf),

      std::tuple<const int *, const int *, int *>(fhalo_start, fhalo_end, fbuf),
      std::tuple<const int *, const int *, int *>(bhalo_start, bhalo_end, dbuf)
  };


  MPI_Request reqs[6]; // 1 for each halo exchange

  int stride_i = 0;
  int stride_j = 0;
  int stride_k = 0;

  




  // for final exchange after everything
  for (const auto & [start_intervals, end_intervals, buf] : intervals)
  {
    if (start_intervals[0] == end_intervals[0]) stride_i = local_N; 
    if (start_intervals[1] == end_intervals[1]) stride_j = local_N;
    if (start_intervals[2] == end_intervals[2]) stride_k = local_N;

    int ii, jj, kk;
    ii = jj = kk = 0;
    // _really_ slow memcpy -- maybe do using openmp?
    for (int i=start_intervals[0];i<end_intervals[0] && ii < local_N;++i, ++ii)
      for (int j=start_intervals[1];j<end_intervals[1] && jj < local_N;++j, ++jj)
        for (int k=start_intervals[2];k<end_intervals[2] && kk < local_N;++k, ++kk)
          buf[kk + local_N * (jj + local_N * ii)] = local_lattice1[kk + stride_k + local_N * (stride_j + jj + local_N * (ii + stride_i))];
    stride_i = stride_j = stride_k = 0;          


    //int MPI_Isendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //              int dest, int sendtag, void *recvbuf, int recvcount,
    //              MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
    //              MPI_Request *request)
    //MPI_Isendrecv(buf, local_N * local_N, MPI_INT, 


  }


  

  

  // TODO
  // - need halo cells
  // - figure out how halo exchange
  // - figure out: single cross-halo for first and second step (9 times) (tbd after halo exchange works) ? 


  // first round: map 4 ranks
  // 0 -- (0, 0, 0)
  // 1 -- (0, 1, 1)
  // 2 -- (1, 0, 1)
  // 3 -- (1, 1, 0)
  //
  // second round: map 4 ranks
  // 0 -- (1, 0, 1)
  // 1 -- (1, 1, 0)
  // 2 -- (0, 0, 1)
  // 3 -- (0, 1, 0)
  
  */
 

  if (rank == 0)
    free(lattice);
  free(local_lattice1);
  free(local_lattice2);
  free(midbuf);
  free(recv_buf);
  MPI_Finalize();
}
