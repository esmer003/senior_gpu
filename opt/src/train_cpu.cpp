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

        a -= LEARNING_RATE * grad_a;
        b -= LEARNING_RATE * grad_b;
        c -= LEARNING_RATE * grad_c;
        d -= LEARNING_RATE * grad_d;
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    printf("CPU Training Time: %lld ms\n", duration.count());

    printf("Learned: a=%.5f b=%.5f c=%.5f d=%.5f\n",
           a, b, c, d);

    return 0;
}