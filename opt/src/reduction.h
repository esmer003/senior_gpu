#ifndef REDUCTION_H
#define REDUCTION_H

#include <cuda_runtime.h>

#ifdef __CUDACC__
// Only the CUDA compiler will see this
__global__ void reduce_sum(float *input, float *partial, int n);
#endif

#endif