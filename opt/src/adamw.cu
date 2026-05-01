/*
grdient descent linear regression

CODE: Training a linear regression model using gradient descent 
GOAL: To learn the parameters w and b of the linear model y_hat = 3x + 2 from 

MSE = 1/n * summation((y - y_pred)^2)
y hat = w * x + b


*/
#include <math.h>
#include "adamw.h"

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