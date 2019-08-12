#include <rtthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "mnist.h"
#include "backend.h"

static void onnx_mnist(int argc, char const *argv[])
{
    int img_index = 0;
    if(argc == 2)
    {
        img_index = atoi(argv[1]);
        if(img_index > TOTAL_IMAGE-1)
        {
            printf("%s: image (0-%d)\n", argv[0], TOTAL_IMAGE-1);
            printf("Maximum datasets available %d\n", TOTAL_IMAGE);
            return;
        }
    }
    else
    {
        printf("%s: image (0-%d)\n", argv[0], TOTAL_IMAGE-1);
        return;
    }
    print_img(img[img_index]);

    // Transpose Input
    int shapeA[] = {1, 28, 28};
    int dimA = 3;
    int perm[] = { 1, 2, 0};
    float* input = transpose(img[img_index], shapeA, dimA, perm);

    // Print input
    // int shapeInput[] = {28, 28, 1};
    // int dimInput = 3;
    // onnx_tensor_info(input, shapeInput, dimInput);

    // 1. Conv2D
    int shapeW3[] = {2, 1, 3, 3};
    int dimW3 = 4;
    int permW3_t[] = { 0, 2, 3, 1};
    float* W3_t = transpose(W3, shapeW3, dimW3, permW3_t);

    float* conv1 = (float*) malloc(sizeof(float)*28*28*2);
    memset(conv1, 0, sizeof(sizeof(float)*28*28*2));
    conv2D(input, 28, 28, 1, W3, 2, 3, 3, 1, 1, 1, 1, B3, conv1, 28, 28);

    free(W3_t);
    free(input);

    // 2. Relu
    relu(conv1, 28*28*2);

    // 3. Maxpool
    float* maxpool1 = (float*) malloc(sizeof(float)*14*14*2);
    memset(maxpool1, 0, sizeof(sizeof(float)*14*14*2));
    maxpool(conv1, 28, 28, 2, 2, 2, 0, 0, 2, 2, 14, 14, maxpool1);

    free(conv1);

    // 4. Conv2D
    int shapeW2[] = {2, 2, 3, 3};
    int dimW2 = 4;
    int perm_t[] = { 0, 2, 3, 1};
    float* W2_t = transpose(W2, shapeW2, dimW2, perm_t);

    float* conv2 = (float*) malloc(sizeof(float)*14*14*2);
    memset(conv2, 0, sizeof(sizeof(float)*14*14*2));
    conv2D(maxpool1, 14, 14, 2, W2_t, 2, 3, 3, 1, 1, 1, 1, B2, conv2, 14, 14);

    free(W2_t);
    free(maxpool1);

    // 5. Relu
    relu(conv2, 14*14*2);

    // 6. Maxpool
    float* maxpool2 = (float*) malloc(sizeof(float)*7*7*2);
    memset(maxpool2, 0, sizeof(sizeof(float)*7*7*2));
    maxpool(conv2, 14, 14, 2, 2, 2, 0, 0, 2, 2, 7, 7, maxpool2);

    free(conv2);

    // Flatten NOT REQUIRED

    // 7. Dense
    int shapeW1[] = {98, 4};
    int dimW1 = 2;
    int permW1_t[] = { 1, 0};
    float* W1_t = transpose(W1, shapeW1, dimW1, permW1_t);

    float* dense1 = (float*) malloc(sizeof(float)*4);
    memset(dense1, 0, sizeof(sizeof(float)*4));
    dense(maxpool2, W1_t, 98, 4, B1, dense1);

    free(W1_t);
    free(maxpool2);

    // 8. Dense
    int shapeW[] = {4, 10};
    int dimW = 2;
    int permW_t[] = { 1, 0};
    float* W_t = transpose(W, shapeW, dimW, permW_t);

    float* dense2 = (float*) malloc(sizeof(float)*10);
    memset(dense2, 0, sizeof(sizeof(float)*10));
    dense(dense1, W_t, 4, 10, B, dense2);

    free(W_t);
    free(dense1);

    // 9. Softmax
    float* output = (float*) malloc(sizeof(float)*10);
    memset(output, 0, sizeof(sizeof(float)*10));
    softmax(dense2, 10, output);

    // 10. Result
    float max = 0;
    int max_index = 0;
    printf("\nPredictions: \n");
    for(int i = 0; i < 10; i++)
    {
        printf("%f ", output[i]);
        if(output[i] > max)
        {
            max = output[i];
            max_index = i;
        }
    }
    printf("\n");
    printf("\nThe number is %d\n", max_index);

    free(dense2);
    free(output);
}
MSH_CMD_EXPORT(onnx_mnist, mnist using onnx backend);
