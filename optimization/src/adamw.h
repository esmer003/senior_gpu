#ifndef ADAMW_H
#define ADAMW_H

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
);

#endif