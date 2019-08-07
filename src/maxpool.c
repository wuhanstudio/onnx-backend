#include "backend.h"

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
             float *output)
{
    int16_t i_ch_in, i_x, i_y;
    int16_t k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                float max = FLT_MIN;
                for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
                {
                    for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            if (input[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)] > max)
                            {
                                max = input[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            }
                        }
                    }
                }
                output[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = max;
            }
        }
    }
}
