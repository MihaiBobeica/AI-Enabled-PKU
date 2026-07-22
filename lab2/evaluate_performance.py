import pandas as pd
import numpy as np
import argparse
import math
from pathlib import Path

# To run this script, in the terminal run the following command in this folder: 
# $ python evaluate_performance.py [input_file] [output_file]
# where input_file should contain the name that all result files in the experiment have in common (so without numbered values), 
# as the script will go through all the ten results automatically and compute the performance metrics

def compute_metrics(filename):
    # Read CSV
    df = pd.read_csv(filename)

    # ---------- Column names ----------
    TIME_COL = "time_s"
    MODE_COL = "mode"
    ALPHA_COL = "alpha_rad"
    PWM_COL = "pwm"

    # Check required columns
    required = [TIME_COL, MODE_COL, ALPHA_COL, PWM_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # -----------------------------------------------------
    # Find the first point where mode == 3 until the end
    # -----------------------------------------------------
    mode = df[MODE_COL].to_numpy()

    stable_index = None

    # for i in range(len(mode)):
    #     if np.all(mode[i:] == 3): #for hardware results
    #     # if np.all(mode[i:-1] == 3): #for software results
    #         stable_index = i
    #         break

    all_abs_alpha = np.abs(df[ALPHA_COL].to_numpy())

    for i in range(len(all_abs_alpha)):
        if np.all(all_abs_alpha[i:] <= math.radians(float(15.0))):
            stable_index = i
            break

    if stable_index is None:
        # raise RuntimeError("Controller never entered permanent PID mode.")
        return (None, None, None, None, None, None)


    stable = df.iloc[stable_index:]

    stable_time = stable.iloc[0][TIME_COL]

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    abs_alpha = np.abs(stable[ALPHA_COL])

    mean_alpha = abs_alpha.mean()
    std_alpha = abs_alpha.std(ddof=0)

    abs_pwm = np.abs(stable[PWM_COL])

    mean_pwm = abs_pwm.mean()
    std_pwm = abs_pwm.std(ddof=0)

    max_overshoot = abs_alpha.max()

    return (stable_time, mean_alpha, std_alpha, mean_pwm, std_pwm, max_overshoot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("output_file", help ="Input name of output folder")
    args = parser.parse_args()

    output_file = args.output_file

    results = []

    for i in range(1, 11):
        input_file = f"{args.input_file}{i:02d}.csv"
        metrics = compute_metrics(input_file)
        results.append((i,) + metrics)

    # Keep only successful runs
    successful = [r[1:] for r in results if r[1] is not None]

    num_success = len(successful)
    num_failed = len(results) - num_success

    if num_success == 0:
        raise RuntimeError("No successful runs.")

    metrics = np.array(successful)

    # Mean over successful runs
    means = np.mean(metrics, axis=0)

    # Standard deviation over successful runs
    stds = np.std(metrics, axis=0, ddof=0)

    with open(output_file, "w") as f:
        for trial, stable_time, mean_alpha, std_alpha, mean_pwm, std_pwm, max_overshoot in results:
            f.write(
                f"{trial} & "
                f"{stable_time:.4f} & "
                f"{mean_alpha:.6f} & "
                f"{std_alpha:.6f} & "
                f"{mean_pwm:.3f} & "
                f"{std_pwm:.3f} & "
                f"{max_overshoot:.6f}\\\\\n"
            )
            
        f.write("\n")

        f.write(f"Successful runs: {num_success}\n")
        f.write(f"Failed runs: {num_failed}\n\n")

        # Summary row in LaTeX format
        f.write(
            "& "
            f"{means[0]:.4f} $\\pm$ {stds[0]:.4f} & "
            f"{means[1]:.6f} $\\pm$ {stds[1]:.6f} & "
            f"{means[2]:.6f} $\\pm$ {stds[2]:.6f} & "
            f"{means[3]:.3f} $\\pm$ {stds[3]:.3f} & "
            f"{means[4]:.3f} $\\pm$ {stds[4]:.3f} & "
            f"{means[5]:.6f} $\\pm$ {stds[5]:.6f}\\\\\n"
        )

    print(f"Results written to {output_file}")
