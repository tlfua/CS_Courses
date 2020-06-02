#include "mpi.h"
#include "reduce.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

int main(int argc, char **argv) {

  int rank; /* rank of process      */
  int npes, n, i, numThread;
  double start, localTime, local_res, global_res;
  float *arr;

  if (argc < 2) {
    std::cerr << "error!! please check the argument" << std::endl;
    return 1;
  }

  // Start up MPI
  MPI_Init(&argc, &argv);
  // Find out process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Find out number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  if (npes != 2)
    MPI_Abort(MPI_COMM_WORLD, 1);

  n = std::stoi(argv[1], nullptr, 0);
  numThread = std::stoi(argv[2], nullptr, 0);

  arr = new (std::nothrow) float[n];

  if (!arr) {
    std::cout << "Memory allocation failed\n";
    return 0;
  }

  for (i = 0; i < n; i++)
    arr[i] = 1.0;

  omp_set_num_threads(numThread);
  // printf("threads:%d\n",numThread);

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  localTime = 0;
  for (i = 0; i < 20; i++) {
  
    local_res = reduce(arr, 0, n);

    MPI_Reduce(&local_res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
  localTime = MPI_Wtime() - start;

  if (rank == 0) {
    printf("%f\n", global_res);
    printf("%f\n", localTime * 50);
  }

  MPI_Finalize();
  return 0;
}
