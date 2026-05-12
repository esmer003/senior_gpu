from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FINAL = BASE.parent / "final"

adamw_file = FINAL / "adamw_synth_log.txt"
sgd_file = FINAL / "sgd_synth_log.txt"

def read_epoch_mse_csv(path):
    rows = []

    encodings_to_try = [
        "utf-16",
        "utf-8",
        "utf-8-sig",
        "latin1"
    ]

    for encoding in encodings_to_try:
        rows.clear()

        try:
            with open(path, "r", encoding=encoding, errors="ignore") as f:
                for line in f:
                    line = line.strip()

                    # Skip empty/header lines
                    if not line:
                        continue

                    if line.lower() == "epoch,mse":
                        continue

                    parts = line.split(",")

                    # Keep only clean epoch,mse rows
                    if len(parts) != 2:
                        continue

                    try:
                        epoch = int(parts[0].strip())
                        mse = float(parts[1].strip())
                        rows.append((epoch, mse))
                    except ValueError:
                        continue

            if rows:
                print(f"Loaded {len(rows)} rows from {path.name} using {encoding}")
                return pd.DataFrame(rows, columns=["epoch", "mse"])

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not parse valid epoch,mse rows from {path}")


adamw = read_epoch_mse_csv(adamw_file)
sgd = read_epoch_mse_csv(sgd_file)

# Keep first 200 epochs for comparison
adamw = adamw[adamw["epoch"] <= 200]
sgd = sgd[sgd["epoch"] <= 200]

plt.figure(figsize=(8, 5))

plt.plot(
    adamw["epoch"],
    adamw["mse"],
    linewidth=2,
    label="AdamW"
)

plt.plot(
    sgd["epoch"],
    sgd["mse"],
    linewidth=2,
    label="SGD"
)

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