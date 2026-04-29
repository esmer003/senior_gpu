#ifndef MODEL_H
#define MODEL_H

#define N 1024
#define BLOCK_SIZE 256
#define EPOCHS 1000
#define LEARNING_RATE 0.001f

// If this is a CUDA kernel, it MUST have __global__
__global__ void gradient_descent(
    const float *X,
    const float *X2, 
    const float *X3,
    const float *Y,
    float *grad_a,
    float *grad_b,
    float *grad_c, 
    float *grad_bias,
    float a,
    float b,
    float c, 
    float bias,
    int n 
); 

#endif