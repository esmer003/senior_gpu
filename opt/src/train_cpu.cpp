#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define N 1024
#define EPOCHS 10000
#define LEARNING_RATE 0.002f

#define TRUE_A 3.0f
#define TRUE_B -2.0f
#define TRUE_C 1.5f
#define TRUE_D 2.0f

void adamw_update(
    float *params,
    float grad,
    float *m,
    float *v,
    float beta1,
    float beta2,
    float weight_decay,
    float lr,
    float eps,
    int timestep
)
{
    *m = beta1 * (*m) + (1.0f - beta1) * grad;
    *v = beta2 * (*v) + (1.0f - beta2) * grad * grad;

    float m_hat = (*m) / (1.0f - powf(beta1, (float)timestep));
    float v_hat = (*v) / (1.0f - powf(beta2, (float)timestep));

    *params -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * (*params));
}
int main()
{
    float x[N];
    float y[N];

    // generate synthetic data
    for (int i = 0; i < N; i++)
    {
        x[i] = -1.0f + 2.0f * i / (N - 1);

        y[i] =
            TRUE_A * x[i] * x[i] * x[i] +
            TRUE_B * x[i] * x[i] +
            TRUE_C * x[i] +
            TRUE_D;
    }

    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    int timestep = 0;

    float m_a = 0.0f, v_a = 0.0f;
    float m_b = 0.0f, v_b = 0.0f;
    float m_c = 0.0f, v_c = 0.0f;
    float m_d = 0.0f, v_d = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float grad_a = 0.0f;
        float grad_b = 0.0f;
        float grad_c = 0.0f;
        float grad_d = 0.0f;

        for (int i = 0; i < N; i++)
        {
            float pred =
                a * x[i] * x[i] * x[i] +
                b * x[i] * x[i] +
                c * x[i] +
                d;

            float error = pred - y[i];

            grad_a += error * x[i] * x[i] * x[i];
            grad_b += error * x[i] * x[i];
            grad_c += error * x[i];
            grad_d += error;
        }

        grad_a /= N;
        grad_b /= N;
        grad_c /= N;
        grad_d /= N;

        timestep++;

        adamw_update(&a, grad_a, &m_a, &v_a, beta1, beta2, weight_decay, LEARNING_RATE, eps, timestep);
        adamw_update(&b, grad_b, &m_b, &v_b, beta1, beta2, weight_decay, LEARNING_RATE, eps, timestep);
        adamw_update(&c, grad_c, &m_c, &v_c, beta1, beta2, weight_decay, LEARNING_RATE, eps, timestep);
        adamw_update(&d, grad_d, &m_d, &v_d, beta1, beta2, weight_decay, LEARNING_RATE, eps, timestep);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    printf("CPU Training Time: %ld ms\n", duration.count());

    printf("Learned: a=%.5f b=%.5f c=%.5f d=%.5f\n",
           a, b, c, d);

    return 0;
}