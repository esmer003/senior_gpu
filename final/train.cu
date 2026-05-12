#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "model.h"
#include "reduction.h"
#include "adamw.h"
#include "data_gen.h"
#include "cuda_check.h"
#include "load_bin_data.h"

// nvcc train.cu model.cu reduction.cu adamw.cpp data_gen.cpp load_bin_data.cpp -o train.exe

static float compute_mse_host(const float* x, const float* y, int n,
                              float a, float b, float c, float d)
{
    float mse = 0.0f;

    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float pred = a * xi * xi * xi + b * xi * xi + c * xi + d;
        float diff = pred - y[i];
        mse += diff * diff;
    }

    return mse / n;
}

int main()
{
    int n = N;
    size_t bytes = n * sizeof(float);

    int batch_size = 256;
    int points_per_batch = batch_size;
    int blocks_per_batch = 1;

    float m_a = 0.0f, v_a = 0.0f;
    float m_b = 0.0f, v_b = 0.0f;
    float m_c = 0.0f, v_c = 0.0f;
    float m_d = 0.0f, v_d = 0.0f;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
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

    bool use_real_data = false;

    if (use_real_data) {
        load_bin_data(h_x, h_y, n);
    } else {
        data_gen(h_x, h_y, n);
    }

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;

    printf("Training: Y = %.1fx^3 + %.1fx^2 + %.1fx + %.1f | N=%d epochs=%d lr=%.4f\n\n",
           TRUE_A, TRUE_B, TRUE_C, TRUE_BIAS, N, EPOCHS, LEARNING_RATE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    FILE* synth_log = NULL;
    if (!use_real_data) {
        synth_log = fopen("adamw_synth_log.txt", "w");
        if (!synth_log) {
            fprintf(stderr, "Failed to open adamw_synth_log.txt for writing.\n");
            return 1;
        }
        fprintf(synth_log, "epoch,mse\n");
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < n; i += points_per_batch) {
            timestep++;

            gradient_descent<<<blocks_per_batch, BLOCK_SIZE>>>(d_x + i,
                                                               d_y + i,
                                                               d_grad_a,
                                                               d_grad_b,
                                                               d_grad_c,
                                                               d_grad_d,
                                                               a,
                                                               b,
                                                               c,
                                                               d,
                                                               points_per_batch);

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_a, d_partial, points_per_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            float grad_a = h_partial[0] / points_per_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_b, d_partial, points_per_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            float grad_b = h_partial[0] / points_per_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_c, d_partial, points_per_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            float grad_c = h_partial[0] / points_per_batch;

            reduce_sum<<<blocks_per_batch, BLOCK_SIZE>>>(d_grad_d, d_partial, points_per_batch);
            CUDA_CHECK(cudaMemcpy(h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost));
            float grad_d = h_partial[0] / points_per_batch;

            adamw_update(&a, grad_a, &m_a, &v_a, beta1, beta2, weight_decay, LEARNING_RATE, eps, timestep);
            adamw_update(&b, grad_b, &m_b, &v_b, beta1, beta2, weight_decay, LEARNING_RATE, eps, timestep);
            adamw_update(&c, grad_c, &m_c, &v_c, beta1, beta2, weight_decay, LEARNING_RATE, eps, timestep);
            adamw_update(&d, grad_d, &m_d, &v_d, beta1, beta2, weight_decay, LEARNING_RATE, eps, timestep);
        }

        if (epoch % 50 == 0 || epoch == EPOCHS - 1) {
            printf("[Epoch %3d] a=%.5f b=%.5f c=%.5f d=%.5f\n", epoch, a, b, c, d);
        }

        if (synth_log) {
            float mse_epoch = compute_mse_host(h_x, h_y, n, a, b, c, d);
            fprintf(synth_log, "%d,%.6f\n", epoch, mse_epoch);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Training time: %.2f ms\n", ms);

    printf("\n---- Results ----\n");
    printf("Learned: a=%.5f b=%.5f c=%.5f d=%.5f\n", a, b, c, d);
    printf("True:    a=%.5f b=%.5f c=%.5f d=%.5f\n", TRUE_A, TRUE_B, TRUE_C, TRUE_BIAS);

    float mse = compute_mse_host(h_x, h_y, n, a, b, c, d);

    if (!use_real_data) {
        printf("Error:   a=%.6f b=%.6f c=%.6f d=%.6f\n",
               fabsf(a - TRUE_A), fabsf(b - TRUE_B),
               fabsf(c - TRUE_C), fabsf(d - TRUE_BIAS));
    }

    if (use_real_data) {
        float y_std = 7.4521f;
        printf("MSE (normalized):     %.6f\n", mse);
        printf("MSE (original units): %.6f\n", mse * y_std * y_std);
        printf("Ojha reference MSE:   15.095050\n");
    } else {
        printf("Final MSE: %.6f\n", mse);
    }

    if (synth_log) {
        fclose(synth_log);
    }

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