#include <cstdio>
#include <cstdlib>
#include "convolution.h"

void Convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    // compute pad number
    std::size_t p = (m - 1)/ 2;

    // construct image_pad
    float* image_pad = (float*)malloc((n + 2*p) * (n + 2*p) * sizeof(float));
    for (std::size_t i = 0; i < (n + 2*p) * (n + 2*p); i++) {
        image_pad[i] = 0;
    }
    for (std::size_t i = 0; i < n*n; i++) {
        std::size_t image_row_index = i/ n;
        std::size_t image_col_index = i % n;
        image_pad[(n + 2*p)*(image_row_index + p) + (image_col_index + p)] = image[i];
    }

    // convolution
    for (std::size_t output_row_index = 0; output_row_index < n; output_row_index++) {
        for (std::size_t output_col_index = 0; output_col_index < n; output_col_index++) {
            output[n*output_row_index + output_col_index] = 0;
            for (std::size_t i = 0; i < m*m; i++) {
                std::size_t mask_row_index = i/ m;
                std::size_t mask_col_index = i % m;
                output[n*output_row_index + output_col_index] += image_pad[(n + 2*p)*(output_row_index+mask_row_index) + (output_col_index+mask_col_index)] * mask[i];
            }
        }
    }

    free(image_pad);
}