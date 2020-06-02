#include "matmul.cuh"

__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              unsigned int n) {
  int block_x = blockIdx.x; // B matrix sub-block column index
  int block_y = blockIdx.y; // A matrix sub-block row index

  int thread_x = threadIdx.x; // the column index in the sub-block
  int thread_y = threadIdx.y; // the row index in the sub-block

  __shared__ int block_dim;
  __shared__ int grid_dim;
  if ((thread_x == 0) && (thread_y == 0)) {
    block_dim = blockDim.x;
    grid_dim = gridDim.x;
  }
  __syncthreads();

  int first_a_first_idx =
      n * block_dim * block_y; // global index to start a sub-matrix a
  int a_step = block_dim;      // global step to jump to next sub-matrix a
  int last_a_first_idx = first_a_first_idx + a_step * (grid_dim - 1);

  int first_b_first_idx =
      block_dim * block_x;    // global index to start a sub-matrix b
  int b_step = block_dim * n; // global step to jump to next sub-matrix b
  int last_b_first_idx = first_b_first_idx + b_step * (grid_dim - 1);

  extern __shared__ float shared_arr[];
  float *shared_a = shared_arr;
  float *shared_b = &shared_a[block_dim * block_dim];
  float sum = 0;
  for (int a_iter = first_a_first_idx, b_iter = first_b_first_idx;
       (a_iter <= last_a_first_idx) && (b_iter <= last_b_first_idx);
       a_iter += a_step, b_iter += b_step) {

    shared_a[thread_y * block_dim + thread_x] = 0;
    shared_b[thread_y * block_dim + thread_x] = 0;

    int A_idx = a_iter + n * thread_y + thread_x;
    if ((0 <= A_idx) && (A_idx < n * n)) {
      shared_a[thread_y * block_dim + thread_x] = A[A_idx];
    }
    int B_idx = b_iter + n * thread_y + thread_x;
    if ((0 <= B_idx) && (B_idx < n * n)) {
      shared_b[thread_y * block_dim + thread_x] = B[B_idx];
    }

    /*
    int A_idx = a_iter + n * thread_y + thread_x;
    int B_idx = b_iter + n * thread_y + thread_x;
    if ((0 <= A_idx) && (A_idx < n * n) && (0 <= B_idx) && (B_idx < n * n)) {
        shared_a[thread_y * block_dim + thread_x] = A[A_idx];
        shared_b[thread_y * block_dim + thread_x] = B[B_idx];
    }
    */
    __syncthreads();

    for (int k = 0; k < block_dim; ++k) {
      sum += shared_a[thread_y * block_dim + k] *
             shared_b[k * block_dim + thread_x];
    }
    __syncthreads();
  }

  // Write the block sub-matrix to global memory
  int C_idx =
      n * block_dim * block_y + block_dim * block_x + n * thread_y + thread_x;
  if ((0 <= C_idx) && (C_idx < n * n)) {
    C[C_idx] = sum;
  }
}

__host__ void matmul(const float *A, const float *B, float *C, unsigned int n,
                     unsigned int block_dim) {
  unsigned int grid_dim =
      ((n % block_dim) == 0) ? (n / block_dim) : (n / block_dim) + 1;

  dim3 block_2D(block_dim, block_dim);
  dim3 grid_2D(grid_dim, grid_dim);

  matmul_kernel<<<grid_2D, block_2D,
                  2 * block_dim * block_dim * sizeof(float)>>>(A, B, C, n);
  cudaDeviceSynchronize();
}
