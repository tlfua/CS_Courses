#include "mpi.h"
#include <iostream>
#include <new>
#include <cstdlib>

void fill_two_arrs(float* arr_A, float* arr_B, int n)
{
    for (int i = 0; i < n; ++i){
        arr_A[i] = i;
	arr_B[i] = n - 1 - i;
    }
}

int main(int argc, char* argv[])
{
    int n = atoi(argv[1]);

    int my_rank;
    int procs = 2;
    //int src, dst;
    int tag = 0;
    float* arr_A = new (std::nothrow) float[n];
    float* arr_B = new (std::nothrow) float[n];
    MPI_Status status;
    double proc_0_start_time, proc_0_end_time;
    double proc_0_duration;
    double proc_1_start_time, proc_1_end_time;
    double proc_1_duration;

    fill_two_arrs(arr_A, arr_B, n);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    if (my_rank == 0) {
	proc_0_start_time = MPI_Wtime();
        
	MPI_Send(arr_A, n, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
	MPI_Recv(arr_B, n, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &status);
	//std::cout << "the last arr element proc_0 recv is " << arr_B[n - 1] << "\n";
	
	proc_0_end_time = MPI_Wtime();
	proc_0_duration = proc_0_end_time - proc_0_start_time;

        MPI_Recv(&proc_1_duration, 1, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD, &status);
	//std::cout << "t0: " << proc_0_duration << "\n";
        //std::cout << "t1: " << proc_1_duration << "\n";
	std::cout << (proc_0_duration + proc_1_duration) * 1000 << "\n";
    } else if (my_rank == 1) {
	proc_1_start_time = MPI_Wtime();
        
	MPI_Recv(arr_A, n, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        //std::cout << "the last arr element proc_1 recv is " << arr_A[n - 1] << "\n";
	MPI_Send(arr_B, n, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
	
	proc_1_end_time = MPI_Wtime();
	proc_1_duration = proc_1_end_time - proc_1_start_time;

        MPI_Send(&proc_1_duration, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    delete[] arr_A;
    delete[] arr_B;
    return 0;
}
