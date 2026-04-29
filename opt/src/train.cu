#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "model.h"
#include "reduction.h"
#include "adamw.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main()
{
    int n = N;
    size_t bytes = n * sizeof(float);

    int batch_size = 256;
    int points_per_batch = batch_size;
    int blocks_per_batch = 1;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    int timestep = 0;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_partial = (float*)malloc(num_blocks * sizeof(float));

    float *d_x;
    float *d_y;
    float *d_partial;
    float *d_grad_a;
    float *d_grad_b;
    float *d_grad_c;
    float *d_grad_d;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_d, bytes));

    // TODO: make sure h_x and h_y are filled before copying.
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;

    float m_a = 0.0f, v_a = 0.0f;
    float m_b = 0.0f, v_b = 0.0f;
    float m_c = 0.0f, v_c = 0.0f;
    float m_d = 0.0f, v_d = 0.0f;

    printf("Training: N=%d epochs=%d lr=%.4f\n\n", N, EPOCHS, LEARNING_RATE);

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        for (int i = 0; i < n; i += points_per_batch)
        {
            timestep++;

            int current_batch = points_per_batch;
            if (i + current_batch > n)
            {
                current_batch = n - i;
            }

            gradient_descent<<<blocks_per_batch, BLOCK_SIZE>>>(
                d_x + i,
                d_x + i,
                d_x + i,
                d_y + i,
                d_grad_a,
                d_grad_b,
                d_grad_c,
                d_grad_d,
                a,
                b,
                c,
                d,
                current_batch
            );

            CUDA_CHECK(cudaDeviceSynchronize());

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_a, d_partial, current_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            float grad_a = h_partial[0] / current_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_b, d_partial, current_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            float grad_b = h_partial[0] / current_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_c, d_partial, current_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            float grad_c = h_partial[0] / current_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_d, d_partial, current_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            float grad_d = h_partial[0] / current_batch;

            m_a = beta1 * m_a + (1.0f - beta1) * grad_a;
            v_a = beta2 * v_a + (1.0f - beta2) * grad_a * grad_a;
            float m_a_hat = m_a / (1.0f - powf(beta1, timestep));
            float v_a_hat = v_a / (1.0f - powf(beta2, timestep));
            a -= LEARNING_RATE * (m_a_hat / (sqrtf(v_a_hat) + eps) + weight_decay * a);

            m_b = beta1 * m_b + (1.0f - beta1) * grad_b;
            v_b = beta2 * v_b + (1.0f - beta2) * grad_b * grad_b;
            float m_b_hat = m_b / (1.0f - powf(beta1, timestep));
            float v_b_hat = v_b / (1.0f - powf(beta2, timestep));
            b -= LEARNING_RATE * (m_b_hat / (sqrtf(v_b_hat) + eps) + weight_decay * b);

            m_c = beta1 * m_c + (1.0f - beta1) * grad_c;
            v_c = beta2 * v_c + (1.0f - beta2) * grad_c * grad_c;
            float m_c_hat = m_c / (1.0f - powf(beta1, timestep));
            float v_c_hat = v_c / (1.0f - powf(beta2, timestep));
            c -= LEARNING_RATE * (m_c_hat / (sqrtf(v_c_hat) + eps) + weight_decay * c);

            m_d = beta1 * m_d + (1.0f - beta1) * grad_d;
            v_d = beta2 * v_d + (1.0f - beta2) * grad_d * grad_d;
            float m_d_hat = m_d / (1.0f - powf(beta1, timestep));
            float v_d_hat = v_d / (1.0f - powf(beta2, timestep));
            d -= LEARNING_RATE * (m_d_hat / (sqrtf(v_d_hat) + eps) + weight_decay * d);
        }

        if (epoch % 50 == 0 || epoch == EPOCHS - 1)
        {
            printf("[Epoch %3d] a=%.5f b=%.5f c=%.5f d=%.5f\n", epoch, a, b, c, d);
        }
    }

    printf("\n---- Results ----\n");
    printf("Learned: a=%.5f b=%.5f c=%.5f d=%.5f\n", a, b, c, d);

    free(h_x);
    free(h_y);
    free(h_partial);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial);
    cudaFree(d_grad_a);
    cudaFree(d_grad_b);
    cudaFree(d_grad_c);
    cudaFree(d_grad_d);

    return 0;
}