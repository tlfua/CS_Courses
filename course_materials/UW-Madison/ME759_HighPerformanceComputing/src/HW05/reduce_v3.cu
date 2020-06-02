#include "reduce.cuh"
#include <iostream>

__global__ void reduce_kernel(const int* g_idata, int* g_odata, unsigned int n)
{ 
    __shared__ unsigned int threads_per_block;
    if (threadIdx.x == 0) {
        threads_per_block = blockDim.x;
    }
    __syncthreads();
    
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = threads_per_block * bid + tid;

    // specially deal with last block
    if ((threadIdx.x == 0) && (blockIdx.x == gridDim.x - 1)) {
        unsigned int last_threads_per_block = n % threads_per_block;
        if (last_threads_per_block != 0) {
            threads_per_block = last_threads_per_block;
        }
    }
    __syncthreads();
    
    extern __shared__ int sh_data[];
    if (tid < threads_per_block) {
        sh_data[tid] = g_idata[global_tid];
    }
    __syncthreads();
    
    // do reduction in shared mem
    /*
    for(unsigned int s = (threads_per_block + 1) / 2; s > 0; s >>= 1) {
        if ((tid < s) && (tid + s < threads_per_block)) {
            sh_data[tid] += sh_data[tid + s];
        }
        __syncthreads();
    }
    */

    unsigned int s;
    unsigned int block_size = threads_per_block;
    while (block_size > 1) {
    
        if ((block_size % 2) == 0) {
	    s = block_size / 2;
	    if (tid < s) {
	        sh_data[tid] += sh_data[tid + s];
	    }
	} else {
	    s = (block_size + 1) / 2;
	    if (tid < s - 1) {
	        sh_data[tid] += sh_data[tid + s];
	    }
	}
	block_size = s;
	__syncthreads();
    }

    /*
    if ((tid == 0) && (bid == gridDim.x - 1)) {
        printf("sh_data:\n");
	for (unsigned int i = 0; i < n; ++i) {
	    printf("%d ", sh_data[i]);
	}
	printf("\n");
    }
    __syncthreads();
    */
    
    // write result for this block to global memory
    if (tid == 0) {
        g_odata[bid] = sh_data[0];
    }
}

__host__ int reduce(const int* arr, unsigned int N, unsigned int threads_per_block)
{
    // arr_size: 32 -> 16 -> 8 -> 4 -> 2 -> 1

    int *idata, *odata;
    cudaMallocManaged(&idata, N * sizeof(int));
    cudaMemcpy(idata, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMallocManaged(&odata, N * sizeof(int));

    // unsigned int idata_size = N;
    unsigned int blocks;
    while (N > 1) {

        blocks = (N + threads_per_block - 1) / threads_per_block;
        reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(int)>>>(idata, odata, N);
        cudaDeviceSynchronize();
    
        /*
	    std::cout << "odata:\n";
	    for (unsigned int i = 0; i < blocks; ++i) {
		    std::cout << odata[i] << ' ';
	    }
        std::cout << '\n';
        */
        
        cudaMemcpy(idata, odata, blocks * sizeof(int), cudaMemcpyDeviceToDevice);
	    N = blocks;
    }

    int res = odata[0];
    cudaFree(idata);
    cudaFree(odata);
    return res;
}
