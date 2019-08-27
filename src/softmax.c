#include "onnx.h"

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

float* softmax_layer(Onnx__GraphProto* graph, const float *input, int64_t* shapeInput, int64_t* shapeOutput, const char* layer_name)
{
    assert(graph != NULL && input != NULL && layer_name != "" && shapeInput[1] > 0);

    float* output = (float*) malloc(sizeof(float)*shapeInput[1]);
    memset(output, 0, sizeof(sizeof(float)*shapeInput[1]));
    softmax(input, shapeInput[1], output);

    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);

    return output;
}
