#include "data_gen.h"

void generate_data(float *h_x, float *h_y, int n)
{
    for (int i = 0; i < n; i++)
    {
        h_x[i] = (float)i / n;

        h_y[i] = TRUE_A * h_x[i] * h_x[i] * h_x[i]
               + TRUE_B * h_x[i] * h_x[i]
               + TRUE_C * h_x[i]
               + TRUE_BIAS;
    }
}