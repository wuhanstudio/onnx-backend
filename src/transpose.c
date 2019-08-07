#include "backend.h"

void onnx_tensor_info(const float* A, int* shape, int dim)
{
    int elem = 1;
    for(int i = 0; i < dim; i++)
    {
        elem = elem * shape[i];
    }

    printf("Array size: %d\n", elem);
    for(int i = 0; i < elem; i++)
    {
        printf( "%f ", A[i] );
        int split = 1;
        for(int j = dim-1; j > 0; j--)
        {
            split = split * shape[j];
            if( (i+1) % split == 0)
            {
                printf("\n");
            }
        }
    }
}

float* transpose(const float* A, int* shape, int dim, int* perm)
{
    // Get array size
    int elem = 1;
    for(int i = 0; i < dim; i++)
    {
        elem = elem * shape[i];
    }

    // Malloc memory for B
    float* B = (float*) malloc(sizeof(float) * elem);
    if(B == NULL)
    {
        return NULL;
    }

    // Malloc memory for shapeB
    int* shapeB = (int*) malloc(sizeof(int) * dim);
    if( shapeB == NULL)
    {
        return NULL;
    }
    for(int i = 0; i < dim; i++)
    {
        shapeB[i] = shape[perm[i]];
    }

    // printf("Shape B: ");
    // for(int i = 0; i < dim; i++)
    // {
    //     printf("%d ", shapeB[i]);
    // }
    // printf("\n");

    // Transpose
    for(int src = 0; src < elem; src++)
    {
        // Get transposed B array
        // A[1][0][3] -> B[3][1][0]
        int temp = src;
        int* indexA = (int*) malloc(sizeof(int) * dim);
        if(indexA == NULL)
        {
            return NULL;
        }
        int* indexB = (int*) malloc(sizeof(int) * dim);
        if(indexB == NULL)
        {
            return NULL;
        }
        for(int i = dim-1; i >= 0; i--)
        {
            indexA[i] = temp % shape[i];
            temp = temp / shape[i];
        }
        for(int i = 0; i < dim; i++)
        {
            indexB[i] = indexA[perm[i]];
        }

        // Get transposed B index 
        // #15 A[1][0][3] -> B[3][1][0] #21
        int dst = 0;
        temp = 1;
        for(int i = dim - 1; i >= 0; i--)
        {
            dst = dst + indexB[i] * temp;
            temp = temp * shapeB[i];
        }

        B[dst] = A[src];

        // printf("#%d: ", src);
        // for(int i = 0; i < dim; i++)
        // {
        //     printf("%d ", indexA[i]);
        // }
        // printf(" -> ");
        // for(int i = 0; i < dim; i++)
        // {
        //     printf("%d ", indexB[i]);
        // }
        // printf("#%d\n", dst);
        free(indexA);
        free(indexB);
    }

    free(shapeB);

    return B;
}
