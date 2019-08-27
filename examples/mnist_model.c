#include <rtthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mnist.h"
#include "onnx.h"

#define MNIST_TEST_IMAGE 1
#define ONNX_MODEL_NAME "mnist-sm.onnx"

#define THREAD_PRIORITY         8
#define THREAD_STACK_SIZE       5120
#define THREAD_TIMESLICE        5

static rt_thread_t tid1 = RT_NULL;

static void mnist_model_entry(void* parameter)
{
    // 0. Load Model
    Onnx__ModelProto* model = onnx_load_model(ONNX_MODEL_NAME);
    if(model == NULL)
    {
        printf("Failed to load model %s\n", ONNX_MODEL_NAME);
        return;
    }

    // 1. Initialize input
    int64_t* shapeInput = (int64_t*) malloc(sizeof(int64_t)*3);
    shapeInput[0] = 28; shapeInput[1] = 28; shapeInput[2] =  1;

    float* input = (float*) malloc(sizeof(int64_t)*28*28);
    memcpy(input, img[MNIST_TEST_IMAGE], sizeof(float)*28*28);

    print_img(input);
    printf("\n");

    // 2. Run Model
    float* output = onnx_model_run(model, input, shapeInput);

    // 3. Print Result
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

    // 4. Free model
    free(shapeInput);
    free(output);
    onnx__model_proto__free_unpacked(model, NULL);
}

static void mnist_model(int argc, char *argv[])
{
    tid1 = rt_thread_create("tonnx_model",
                    mnist_model_entry, RT_NULL,
                    THREAD_STACK_SIZE,
                    THREAD_PRIORITY, THREAD_TIMESLICE);

    if (tid1 != RT_NULL)
    {

        rt_thread_startup(tid1);
    }
    else
    {
        rt_kprintf("Failed to start onnx thread\n");
    }
}
MSH_CMD_EXPORT(mnist_model, load mnist onnx model from file);
