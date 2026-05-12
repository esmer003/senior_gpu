import optuna
import subprocess
import re

EXECUTABLE = "./gpu_train"

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    result = subprocess.run(
        [EXECUTABLE, str(lr)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("CUDA training failed")

    match = re.search(r"FINAL_LOSS\s+([0-9.eE+-]+)", result.stdout)

    if not match:
        print(result.stdout)
        raise RuntimeError("FINAL_LOSS not found")

    return float(match.group(1))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("\nBest Optuna Result")
print("Best loss:", study.best_value)
print("Best learning rate:", study.best_params["lr"])