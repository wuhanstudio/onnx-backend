#include <rtthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mnist.h"
#include "onnx.h"

#define MNIST_TEST_IMAGE 1
#define ONNX_MODEL_NAME "/mnist-sm.onnx"

#define THREAD_PRIORITY         8
#define THREAD_STACK_SIZE       5120
#define THREAD_TIMESLICE        5

static rt_thread_t tid1 = RT_NULL;

static void mnist_sm_entry(void* parameter)
{
    // Load Model
    Onnx__ModelProto* model = onnx_load_model(ONNX_MODEL_NAME);
    if(model == NULL)
    {
        printf("Failed to load model %s\n", ONNX_MODEL_NAME);
        return;
    }

    // Set input image: NWHC
    print_img(img[MNIST_TEST_IMAGE]);

    // 0. Initialize input shape
    int64_t* shapeInput = (int64_t*) malloc(sizeof(int64_t)*3);
    int64_t* shapeOutput = (int64_t*) malloc(sizeof(int64_t)*3);
    shapeInput[0] = 28;
    shapeInput[1] = 28;
    shapeInput[2] =  1;

    // 1. Transpose
    // float* input = transpose_layer(model->graph, img[img_index], shapeInput, shapeOutput, "Transpose6");
    // memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);

    // 2. Conv2D
    float* conv1 = conv2D_layer(model->graph, img[MNIST_TEST_IMAGE], shapeInput, shapeOutput, "conv2d_5");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    // free(input);

    // 3. Relu
    float* relu1 = relu_layer(model->graph, conv1, shapeInput, shapeOutput, "Relu1");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(conv1);

    // 4. Maxpool
    float* maxpool1 = maxpool_layer(model->graph, relu1, shapeInput, shapeOutput, "max_pooling2d_5");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(relu1);

    // 5. Conv2D
    float* conv2 = conv2D_layer(model->graph, maxpool1, shapeInput, shapeOutput, "conv2d_6");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(maxpool1);

    // 6. Relu
    float* relu2 = relu_layer(model->graph, conv2, shapeInput, shapeOutput, "Relu");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(conv2);

    // 7. Maxpool
    float* maxpool2 = maxpool_layer(model->graph, relu2, shapeInput, shapeOutput, "max_pooling2d_6");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(relu2);

    // 8. Transpose
    // float* maxpool2_t = transpose_layer(model->graph, maxpool2, shapeInput, shapeOutput, "Transpose1");
    // memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    // free(maxpool2);

    // 9. Flatten
    shapeInput[1] = shapeInput[0] * shapeInput[1] * shapeInput[2]; 
    shapeInput[2] = 1; 
    shapeInput[0] = 1; 

    // 10. Dense
    float* matmul1 = matmul_layer(model->graph, maxpool2, shapeInput, shapeOutput, "dense_5");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(maxpool2);

    // 11. Add
    float* dense1 = add_layer(model->graph, matmul1, shapeInput, shapeOutput, "Add1");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(matmul1);

    // 12. Dense
    float* matmul2 = matmul_layer(model->graph, dense1, shapeInput, shapeOutput, "dense_6");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(dense1);

    // 13. Add
    float* dense2 = add_layer(model->graph, matmul2, shapeInput, shapeOutput, "Add");
    memcpy(shapeInput, shapeOutput, sizeof(int64_t)*3);
    free(matmul2);

    // 14. Softmax
    float* output = softmax_layer(model->graph, dense2, shapeInput, shapeOutput, "Softmax");
    free(dense2);

    // 15. Identity
    // Do Nothing Here

    // Result
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

    // Free model
    free(shapeInput);
    free(shapeOutput);
    free(output);
    onnx__model_proto__free_unpacked(model, NULL);

    return;
}

static void mnist_sm(int argc, char const *argv[])
{

    tid1 = rt_thread_create("tmnist_sm",
                    mnist_sm_entry, RT_NULL,
                    THREAD_STACK_SIZE,
                    THREAD_PRIORITY, THREAD_TIMESLICE);

    if (tid1 != RT_NULL)
    {

        rt_thread_startup(tid1);
    }
    else
    {
        rt_kprintf("Failed to start mnist-sm thread\n");
    }

}
MSH_CMD_EXPORT(mnist_sm, mnist small model)
