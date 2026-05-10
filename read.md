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