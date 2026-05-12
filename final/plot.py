import pandas as pd
import matplotlib.pyplot as plt

adamw = pd.read_csv("adamw_synth_log.txt")
sgd = pd.read_csv("sgd_synth_log.txt")

plt.figure(figsize=(8,5))

plt.plot(adamw["epoch"], adamw["mse"], label="AdamW")
plt.plot(sgd["epoch"], sgd["mse"], label="SGD")

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Synthetic Dataset: MSE vs Epoch")

plt.legend()
plt.grid(True)

plt.savefig("synthetic_mse_plot.png", dpi=300)
plt.show()