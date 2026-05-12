from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FINAL = BASE.parent / "final"

adamw_file = FINAL / "adamw_synth_log.txt"
sgd_file = FINAL / "sgd_synth_log.txt"

adamw = pd.read_csv(adamw_file, encoding="utf-16")
sgd = pd.read_csv(sgd_file, encoding="utf-16")

adamw = adamw[adamw["epoch"] <= 200]
sgd = sgd[sgd["epoch"] <= 200]

plt.figure(figsize=(8, 5))
plt.plot(adamw["epoch"], adamw["mse"], linewidth=2, label="AdamW")
plt.plot(sgd["epoch"], sgd["mse"], linewidth=2, label="SGD")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE (log scale)")
plt.title("Synthetic Cubic: MSE vs Epoch")
plt.legend()
plt.grid(True, alpha=0.3)

out = BASE / "synthetic_mse_vs_epoch_log.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved plot to {out}")