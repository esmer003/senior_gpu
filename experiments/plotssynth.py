from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FINAL = BASE.parent / "final"

adamw_file = FINAL / "adamw_synth_log.txt"
sgd_file = FINAL / "sgd_synth_log.txt"


def read_mse_log(path):
    rows = []
    mse_log_pattern = re.compile(
        r"MSE_LOG\s+(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )

    encodings = ["utf-16", "utf-8-sig", "utf-8", "latin1"]

    for encoding in encodings:
        rows.clear()

        try:
            with open(path, "r", encoding=encoding, errors="ignore") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    match = mse_log_pattern.search(line)
                    if match:
                        rows.append((int(match.group(1)), float(match.group(2))))
                        continue

                    if line.lower() == "epoch,mse":
                        continue

                    parts = line.split(",")
                    if len(parts) == 2:
                        try:
                            rows.append((int(parts[0].strip()), float(parts[1].strip())))
                        except ValueError:
                            continue

            if rows:
                print(f"Loaded {len(rows)} rows from {path.name} using {encoding}")
                return pd.DataFrame(rows, columns=["epoch", "mse"])

        except UnicodeError:
            continue

    raise ValueError(f"Could not parse valid MSE rows from {path}")


adamw = read_mse_log(adamw_file)
sgd = read_mse_log(sgd_file)

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