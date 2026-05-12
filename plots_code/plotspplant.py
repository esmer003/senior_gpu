from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FINAL = BASE.parent / "final"

adamw_file = FINAL / "adamw_pplant_log.txt"
sgd_file = FINAL / "sgd_pplant_log.txt"

adamw = pd.read_csv(adamw_file)
sgd = pd.read_csv(sgd_file)

adamw = adamw[adamw["epoch"] <= 200]
sgd = sgd[sgd["epoch"] <= 200]

# Full-scale plot
plt.figure(figsize=(8, 5))

plt.plot(adamw["epoch"], adamw["mse"], linewidth=2, label="AdamW")
plt.plot(sgd["epoch"], sgd["mse"], linewidth=2, label="SGD")

plt.axhline(15.095050, linestyle="--", label="Ojha Reference MSE")

plt.xlabel("Epoch")
plt.ylabel("MSE (original units)")
plt.title("Power Plant: MSE vs Epoch")

plt.xlim(0, 200)

plt.legend()
plt.grid(True, alpha=0.3)

full_out = BASE / "powerplant_mse_vs_epoch_full.png"

plt.savefig(full_out, dpi=150, bbox_inches="tight")
plt.show()

# Zoomed plot
plt.figure(figsize=(8, 5))

plt.plot(adamw["epoch"], adamw["mse"], linewidth=2, label="AdamW")
plt.plot(sgd["epoch"], sgd["mse"], linewidth=2, label="SGD")

plt.axhline(15.095050, linestyle="--", label="Ojha Reference MSE")

plt.xlabel("Epoch")
plt.ylabel("MSE (original units)")
plt.title("Power Plant: MSE vs Epoch, Zoomed")

plt.xlim(0, 200)
plt.ylim(15.0, 16.8)

plt.legend()
plt.grid(True, alpha=0.3)

zoom_out = BASE / "powerplant_mse_vs_epoch_zoomed.png"

plt.savefig(zoom_out, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved: {full_out}")
print(f"Saved: {zoom_out}")