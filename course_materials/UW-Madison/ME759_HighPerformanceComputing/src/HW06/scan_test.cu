#include "scan.cuh"
#include <iostream>

// 1.
__global__ void hillis_steele(const float *g_idata, float *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int threads_per_block = blockDim.x;
    unsigned int g_tid = threads_per_block * bid + tid;
    
    extern volatile __shared__  float temp[];
    if (g_tid < n) {
        temp[tid] = g_idata[g_tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    unsigned int pout = 0, pin = 1;
    for(unsigned int offset = 1; offset < threads_per_block; offset *= 2) {
        // swap
        pout = 1 - pout;
        pin = 1 - pout;

        if (tid >= offset) {
            temp[pout * threads_per_block + tid] = temp[pin * threads_per_block + tid] + temp[pin * threads_per_block + tid - offset];
        } else {
            temp[pout * threads_per_block + tid] = temp[pin * threads_per_block + tid];
        }
        __syncthreads(); // I need this here before I start next iteration 
    }
    
    g_odata[g_tid] = temp[pout * threads_per_block + tid];
    /*
    printf("%f ", g_odata[tid]);
    __syncthreads();
    */
}

// 2.
__global__ void extract_sums(const float *g_elems, float *g_sums, unsigned int step)
{
    unsigned int tid = threadIdx.x;
    g_sums[tid] = g_elems[step * tid + (step - 1)]; // the last elem of a block in step 1.
}

// 4.
__global__ void add_scanned_sums(float *g_elems, float *g_sums, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int threads_per_block = blockDim.x;
    unsigned int g_tid = threads_per_block * bid + tid;

    if (bid == 0) {
        return;
    }

    __shared__ float sh_sum;
    if (tid == 0) {
        sh_sum = g_sums[bid - 1];
    }
    __syncthreads();

    g_elems[g_tid] += sh_sum;
}

// 5.
/*
__host__ void shift(const float *g_elems, float *out, unsigned int n)
{
    out[0] = 0;
    if (n == 1) {
        return;
    }
    cudaMemcpy(out + 1, g_elems, (n - 1) * sizeof(float), cudaMemcpyDeviceToHost);
}
*/

__host__ void scan(const float* in, float* out, unsigned int n, unsigned int threads_per_block)
{
    out[0] = 0;
    if (n == 1) {
        return;
    }

    float *g_elems;
    cudaMallocManaged(&g_elems, n * sizeof(float));
    cudaMemcpy(g_elems, in, n * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    // 1. scan elements per block
    hillis_steele<<<blocks, threads_per_block, 2 * threads_per_block * sizeof(float)>>>(g_elems, g_elems, n);
    cudaDeviceSynchronize();

    std::cout << "After 1, g_elems\n";
    for (unsigned int i = 0; i < n; ++i) {
        std::cout << g_elems[i] << " ";
    }
    std::cout << "\n";

    if (blocks == 1) {
        cudaMemcpy(out + 1, g_elems, (n - 1) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(g_elems);
        return;
    }

    float *g_sums;
    cudaMallocManaged(&g_sums, (blocks - 1) * sizeof(float));
    // 2. extract sums
    extract_sums<<<1, (blocks - 1)>>>(g_elems, g_sums, threads_per_block);
    cudaDeviceSynchronize();

    std::cout << "After 2, g_sums\n";
    for (unsigned int i = 0; i < blocks - 1; ++i) {
        std::cout << g_sums[i] << " ";
    }
    std::cout << "\n";

    // 3. scan sums
    hillis_steele<<<1, threads_per_block, 2 * threads_per_block * sizeof(float)>>>(g_sums, g_sums, blocks - 1);
    cudaDeviceSynchronize();

    std::cout << "After 3, g_sums\n";
    for (unsigned int i = 0; i < blocks - 1; ++i) {
        std::cout << g_sums[i] << " ";
    }
    std::cout << "\n";

    // 4. add scanned sums
    add_scanned_sums<<<blocks, threads_per_block>>>(g_elems, g_sums, n);
    cudaDeviceSynchronize();

    std::cout << "After 4, g_elems\n";
    for (unsigned int i = 0; i < n; ++i) {
        std::cout << g_elems[i] << " ";
    }
    std::cout << "\n";

    // 5. shift right by one
    cudaMemcpy(out + 1, g_elems, (n - 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(g_elems);
    cudaFree(g_sums);
}
