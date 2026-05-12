#include <stdio.h>
#include <cuda_runtime.h>
#include "model.h"

__global__ void gradient_descent(
    const float * X,
    const float * Y,
    float *grad_a,
    float *grad_b,
    float *grad_c, 
    float *grad_bias,
    float a,
    float b,
    float c, 
    float bias,
    int n 
)
{
    // 1. Correct global thread ID calculation
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    // 2. Boundary check: ensure we don't access memory out of bounds
    if(i >= n) return; 

    // 3. Logic: Calculate powers of x locally
    float x = X[i];
    float x2 = x * x;
    float x3 = x2 * x;

    // 4. FIX: Use 'bias' instead of 'd'
    // Calculation: y_hat = a*x^3 + b*x^2 + c*x + bias
    float y_hat = a * x3 + b * x2 + c * x + bias; 
    
    // 5. Calculate error (Difference between prediction and ground truth)
    float error = y_hat - Y[i]; 

    // 6. Calculate Gradients (Partial derivatives of the loss function)
    grad_a[i] = 2.0f * error * x3;
    
    // 7. FIX: Removed [i] from x2. x2 is a local float (scalar), not a pointer.
    grad_b[i] = 2.0f * error * x2;
    
    grad_c[i] = 2.0f * error * x;
    grad_bias[i] = 2.0f * error;
}