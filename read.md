senior project
## CPU vs GPU Results

The model was trained on synthetic cubic data:

Y = 3.0x^3 - 2.0x^2 + 1.5x + 2.0

### GPU Result
- Time: 3743.2117 ms
- Learned: a=2.99388, b=-2.00072, c=1.50024, d=1.99898

### CPU Result
- Time: 121 ms
- Learned: a=2.99664, b=-1.99939, c=1.50176, d=1.99969

### Conclusion
Both CPU and GPU versions successfully learned the cubic function. For this small dataset, the CPU version was faster because the GPU has overhead from kernel launches, memory transfers, and synchronization. The GPU implementation is still useful because it demonstrates CUDA parallelization and would scale better for larger workloads.

# GPU-Accelerated Cubic Regression with AdamW

## Overview

This project explores CPU/GPU coordination and CUDA-based acceleration for machine learning training workloads. A cubic regression model was implemented and trained using gradient descent with the AdamW optimizer.

The project compares:
- CPU training
- GPU training using CUDA
- Hyperparameter search using multiple learning rates

The goal was to investigate optimization behavior, convergence, and training performance across CPU and GPU implementations.

---

# Synthetic Training Data

The model was trained on synthetic cubic data generated from:

\[
Y = 3.0x^3 - 2.0x^2 + 1.5x + 2.0
\]

Dataset size:
- N = 1024 samples

Training settings:
- Epochs = 10000
- Optimizer = AdamW

---

# CUDA Features Used

- CUDA kernels
- Parallel gradient computation
- GPU reduction kernels
- CUDA memory management
- CUDA timing events
- CPU/GPU comparison

---

# CPU Hyperparameter Search Results

| Learning Rate | Final Loss | Time (ms) |
|---|---|---|
| 0.0005 | 0.005177 | 182 |
| 0.0010 | 0.000020 | 184 |
| 0.0020 | 0.000000 | 187 |
| 0.0050 | 0.000000 | 189 |

Best CPU learning rate:
- 0.0050

Final learned parameters:

```text
a=2.99832
b=-1.99966
c=1.50086
d=1.99983
```

---

# GPU Hyperparameter Search Results

| Learning Rate | Final Loss | Time (ms) |
|---|---|---|
| 0.0005 | 0.000819 | 3598.86 |
| 0.0010 | 0.000819 | 3598.77 |
| 0.0020 | 0.000819 | 3624.36 |
| 0.0050 | 0.000819 | 3881.78 |

Best GPU learning rate:
- 0.0005

Final learned parameters:

```text
a=2.99425
b=-2.00085
c=1.49994
d=1.99906
```

---

# Conclusions

Both CPU and GPU implementations successfully learned the cubic function with very low error.

For this small dataset size, the CPU implementation was faster because GPU execution includes:
- kernel launch overhead
- memory transfer overhead
- synchronization costs

However, the GPU implementation demonstrates:
- CUDA parallel programming
- GPU-accelerated optimization
- parallel reduction techniques
- scalable training infrastructure

The project successfully demonstrates machine learning optimization using CUDA and CPU/GPU coordination concepts.
