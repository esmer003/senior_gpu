#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "reduction.h"


__global__ void reduce_sum(float *input, float *partial, int n)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n)
        sdata[tid] = input[idx];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}