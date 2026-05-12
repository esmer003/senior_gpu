#ifndef REDUCTION_H
#define REDUCTION_H

#include <cuda_runtime.h>

__global__ void reduce_sum(float *input, float *partial, int n);

#endif