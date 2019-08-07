#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include <rtthread.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>

void onnx_tensor_info(const float* A, int* shape, int dim);
float* transpose(const float* A, int* shape, int dim, int* perm);

void conv2D(const float *input,                                                // input image
            const uint16_t dim_im_in_x,                                        // input image dimention x
            const uint16_t dim_im_in_y,                                        // input image dimention y
            const uint16_t ch_im_in,                                           // number of input image channels
            const float *weight,                                               // kernel weights
            const uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
            const uint16_t dim_kernel_x,                                       // filter kernel size x
            const uint16_t dim_kernel_y,                                       // filter kernel size y
            const uint16_t padding_x,                                          // padding sizes x
            const uint16_t padding_y,                                          // padding sizes y
            const uint16_t stride_x,                                           // stride x
            const uint16_t stride_y,                                           // stride y
            const float *bias,                                                 // bias
            float *output,                                                     // output image
            const uint16_t dim_im_out_x,                                       // output image dimension x
            const uint16_t dim_im_out_y                                        // output image dimension y
);

void relu(float *data, uint32_t size);

void maxpool(const float *input,
             const uint16_t dim_im_in_x,  // input image dimension x or W
             const uint16_t dim_im_in_y,  // input image dimension y or H
             const uint16_t ch_im_in,     // number of input image channels
             const uint16_t dim_kernel_x, // window kernel size
             const uint16_t dim_kernel_y, // window kernel size
             const uint16_t padding_x,    // padding sizes
             const uint16_t padding_y,    // padding sizes
             const uint16_t stride_x,     // stride
             const uint16_t stride_y,     // stride
             const uint16_t dim_im_out_x, // output image dimension x or W
             const uint16_t dim_im_out_y, // output image dimension y or H
             float *output);

void dense(const float *input,              // pointer to vector
           const float *weight,             // pointer to matrix
           const uint16_t dim_vec,         // length of the vector
           const uint16_t num_of_rows,     // numCol of A
           const float *bias,
           float *output);

void softmax(const float *input, const uint32_t dim_vec, float *output);

#endif // __TRANSPOSE_H__
