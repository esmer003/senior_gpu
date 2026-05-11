## Optimizing CPU and GPU Coordination for Training
This project delves into how we can better use GPU lead machine learning, using CUDA and compares the performance between CPU and CPU. The project focus changed from it's original focus do to constraints but the current project focuses on imprving traning with hyperparameter optimization. 


Our work implemets: 
- CUDA computation
- Parallel reduction kernels
- AdamW optimization
- Automated hyperparameter optimization using Optuna

## Project Structure
```text
opt/src/
│
├── train.cu              # GPU training pipeline
├── train_cpu.cpp         # CPU training implementation
├── model.cu              # CUDA gradient kernels
├── reduction.cu          # Parallel reduction kernels
├── adamw.cu              # AdamW optimizer
├── data_gen.cu           # Synthetic data generation
├── optuna_search.py      # Optuna hyperparameter optimization
│
├── model.h
├── reduction.h
├── adamw.h
└── data_gen.h
```