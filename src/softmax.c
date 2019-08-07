#include "backend.h"

void softmax(const float *input, const uint32_t dim_vec, float *output)
{
    float sum = 0.0f;

    for(int i = 0; i < dim_vec; i++)
    {
        output[i] = expf(input[i]);
        sum = sum + output[i];
    }

    for(int i = 0; i < dim_vec; i++)
    {
        output[i] = output[i] / sum;
    }
}
