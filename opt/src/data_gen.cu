#define TRUE_A 3.0f
#define TRUE_B -2.0f
#define TRUE_C 1.5f
#define TRUE_BIAS 2.0f

//generate synthetic data for training
    for (int i = 0; i < n; i++)
    {
        h_x[i] = (float)i / n;
        // Cubic: y = 3x^3 - 2x^2 + 1.5x + 2
        h_y[i] = 3.0f * h_x[i] * h_x[i] * h_x[i]
                - 2.0f * h_x[i] * h_x[i]
                + 1.5f * h_x[i]
                + 2.0f;
}