#include "stencil.cuh"
#include <iostream>

__global__ void stencil_kernel(const float *image, const float *mask,
                               float *output, unsigned int n, unsigned int R) {
  extern __shared__ float shared_arr[];
  float *shared_image_pad = shared_arr;
  float *shared_mask = &shared_image_pad[blockDim.x];
  float *shared_output = &shared_mask[2 * R + 1];

  // assign shared_mask
  if (threadIdx.x < 2 * R + 1) {
    shared_mask[threadIdx.x] = mask[threadIdx.x];
  }
  __syncthreads();

  // assign general block_pixels
  __shared__ unsigned int block_pixels;
  if (threadIdx.x == 0) {
    block_pixels = blockDim.x - 2 * R;
  }
  __syncthreads();

  // specially deal with last block
  if ((threadIdx.x == 0) && (blockIdx.x == gridDim.x - 1)) {
    unsigned int last_block_pixels = n % block_pixels;
    if (last_block_pixels != 0) {
      block_pixels = last_block_pixels;
    }
  }
  __syncthreads();

  // assign shared_image_pad
  int global_image_idx;
  if (threadIdx.x < R) {
    global_image_idx = (blockDim.x - 2 * R) * blockIdx.x + (threadIdx.x - R);
    if (global_image_idx < 0) {
      shared_image_pad[threadIdx.x] = 0;
    } else {
      shared_image_pad[threadIdx.x] = image[global_image_idx];
    }
  } else if ((R <= threadIdx.x) && (threadIdx.x < R + block_pixels)) {
    global_image_idx = (blockDim.x - 2 * R) * blockIdx.x + (threadIdx.x - R);
    shared_image_pad[threadIdx.x] = image[global_image_idx];
  } else if ((R + block_pixels <= threadIdx.x) &&
             (threadIdx.x < 2 * R + block_pixels)) {
    global_image_idx = (blockDim.x - 2 * R) * (blockIdx.x + 1) +
                       (threadIdx.x - R - block_pixels);
    if (global_image_idx >= (int)n) {
      shared_image_pad[threadIdx.x] = 0;
    } else {
      shared_image_pad[threadIdx.x] = image[global_image_idx];
    }
  }
  __syncthreads();

  // convolution
  if (threadIdx.x < block_pixels) {
    float sum = 0;
    for (unsigned int i = 0; i < 2 * R + 1; ++i) {
      sum += shared_mask[i] * shared_image_pad[threadIdx.x + i];
    }
    shared_output[threadIdx.x] = sum;
  }
  __syncthreads();

  if (threadIdx.x < block_pixels) {
    output[(blockDim.x - 2 * R) * blockIdx.x + threadIdx.x] =
        shared_output[threadIdx.x];
  }
}

/*
__global__ void stencil_kernel_test(const float* image, const float* mask,
float* output, unsigned int n, unsigned int R,\ int *b_pixels, float
*b_image_pad)
{
    extern __shared__ float shared_arr[];
    float *shared_image_pad = shared_arr;
    float *shared_mask = &shared_image_pad[blockDim.x];
    float *shared_output = &shared_mask[2 * R + 1];

    // assign shared_mask
    if (threadIdx.x < 2 * R + 1) {
        shared_mask[threadIdx.x] = mask[threadIdx.x];
    }
    __syncthreads();

    // assign general block_pixels
    __shared__ unsigned int block_pixels;
    if (threadIdx.x == 0) {
        block_pixels = blockDim.x - 2 * R;
    }
    __syncthreads();

    // specially deal with last block
    if ((threadIdx.x == 0) && (blockIdx.x == gridDim.x - 1)) {
        unsigned int last_block_pixels = n % block_pixels;
        if (last_block_pixels != 0) {
            block_pixels = last_block_pixels;
        }
    }
    if (threadIdx.x == 0) {
        b_pixels[blockIdx.x] = block_pixels;
    }
    __syncthreads();

    // assign shared_image_pad
    int global_thread_idx;
    if (threadIdx.x < R) {
        global_thread_idx = (blockDim.x - 2 * R) * blockIdx.x - R + threadIdx.x;
        if (global_thread_idx < 0) {
            shared_image_pad[threadIdx.x] = 0;
        } else {
            shared_image_pad[threadIdx.x] = image[global_thread_idx];
        }
    } else if ((R <= threadIdx.x) && (threadIdx.x < R + block_pixels)) {
        global_thread_idx = (blockDim.x - 2 * R) * blockIdx.x - R + threadIdx.x;
        shared_image_pad[threadIdx.x] = image[global_thread_idx];
    } else if ((R + block_pixels <= threadIdx.x) && (threadIdx.x < 2 * R +
block_pixels)) { global_thread_idx = (blockDim.x - 2 * R) * (blockIdx.x + 1) +
(threadIdx.x - R - block_pixels); if (global_thread_idx >= (int)n) {
            shared_image_pad[threadIdx.x] = 0;
        } else {
            shared_image_pad[threadIdx.x] = image[global_thread_idx];
        }

        //shared_image_pad[threadIdx.x] = 0;
    }
    __syncthreads();
    b_image_pad[blockDim.x * blockIdx.x + threadIdx.x] =
shared_image_pad[threadIdx.x];

    // convolution
    if (threadIdx.x < block_pixels) {
        float sum = 0;
        for (unsigned int i = 0; i < 2 * R + 1; ++i) {
            sum += shared_mask[i] * shared_image_pad[threadIdx.x + i];
        }
        shared_output[threadIdx.x] = sum;
    }
    __syncthreads();

    if (threadIdx.x < block_pixels) {
        output[(blockDim.x - 2 * R) * blockIdx.x + threadIdx.x] =
shared_output[threadIdx.x];
    }
}
*/

__host__ void stencil(const float *image, const float *mask, float *output,
                      unsigned int n, unsigned int R,
                      unsigned int threads_per_block) {
  int shared_image_pad_len = threads_per_block;
  int shared_mask_len = 2 * R + 1;
  int shared_output_len = threads_per_block - 2 * R;
  int total_len = shared_image_pad_len + shared_mask_len + shared_output_len;
  int blocks;
  if (n % shared_output_len == 0) {
    blocks = n / shared_output_len;
  } else {
    blocks = n / shared_output_len + 1;
  }

  stencil_kernel<<<blocks, threads_per_block, total_len * sizeof(float)>>>(
      image, mask, output, n, R);
  cudaDeviceSynchronize();

  /*
  int *b_pixels;
  cudaMallocManaged(&b_pixels, blocks * sizeof(int));
  float *b_image_pad;
  cudaMallocManaged(&b_image_pad, blocks * threads_per_block * sizeof(float));

  stencil_kernel_test<<<blocks, threads_per_block, total_len *
  sizeof(float)>>>(image, mask, output, n, R,\ b_pixels, b_image_pad);
  cudaDeviceSynchronize();

  for (int i = 0; i < blocks; ++i) {
      std::cout << "block[" << i << "]\n";

      std::cout << "b_pixels = " << b_pixels[i] << "\n";

      std::cout << "b_image_pad = ";
      for (unsigned int j = 0; j < threads_per_block; ++j) {
          std::cout << "[" << j << "]" << b_image_pad[threads_per_block * i + j]
  << " ";
      }
      std::cout << "\n";

  }

  cudaFree(b_pixels);
  cudaFree(b_image_pad);
  */
}
