## Optimizing CPU and GPU Coordination for Training
This project delves into how we can better use GPU lead machine learning, using CUDA and compares the performance between CPU and CPU. The project focus changed from it's original focus do to constraints but the current project focuses on imprving traning with hyperparameter optimization. 

Our work implemets: 
- CUDA computation
- Parallel reduction kernels
- AdamW optimization
- CPU and GPU training comparisons
- Automated hyperparameter optimization using Optuna

## Repo Structure
```text
senior_gpu/
├── cuda_examples/      # Small CUDA practice/example programs
├── final/              # Main finalized CUDA training implementation
├── optimization/       # Hyperparameter optimization framework
│   └── src/
│       ├── train.cu
│       ├── train_cpu.cpp
│       ├── model.cu
│       ├── reduction.cu
│       ├── adamw.cu
│       ├── data_gen.cu
│       └── optuna_search.py
├── experiments/        # Experimental scripts and benchmarking code
├── data/               # Dataset binary files
├── artifacts/
│   ├── logs/           # Training logs
│   └── plots/          # Generated plots and visualizations
├── build/              # Local compiled outputs
├── read.md
└── .gitignore
```