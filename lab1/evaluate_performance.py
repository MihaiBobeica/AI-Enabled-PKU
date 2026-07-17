import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# To run this script, in the terminal run the following command in this folder: 
# $ python evaluate_performance.py [filename] [output_dir]
# where filename should contain the name that all result files in the experiment have in common (so without numbered values), 
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

    for i in range(len(mode)):
        if np.all(mode[i:] == 3):
            stable_index = i
            break

    if stable_index is None:
        raise RuntimeError("Controller never entered permanent PID mode.")

    stable = df.iloc[stable_index:]

    stable_time = stable.iloc[0][TIME_COL]

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    abs_alpha = np.abs(stable[ALPHA_COL])

    mean_alpha = abs_alpha.mean()
    std_alpha = abs_alpha.std(ddof=1)

    abs_pwm = np.abs(stable[PWM_COL])

    mean_pwm = abs_pwm.mean()
    std_pwm = abs_pwm.std(ddof=1)

    max_overshoot = abs_alpha.max()

    # base_name = Path(filename).stem
    # output_file = Path(output_dir) / f"{base_name}_metrics.txt"

    # with open(output_file, "w") as f:
    #     f.write("===== Evaluation Metrics =====\n\n")
    #     f.write(f"Input file           : {filename}\n")
    #     f.write(f"Stable time          : {stable_time:.4f} s\n")
    #     f.write(f"Mean |alpha|         : {mean_alpha:.6f} rad ({np.degrees(mean_alpha):.3f} deg)\n")
    #     f.write(f"Std |alpha|          : {std_alpha:.6f} rad ({np.degrees(std_alpha):.3f} deg)\n")
    #     f.write(f"Mean |PWM|           : {mean_pwm:.3f}\n")
    #     f.write(f"Std |PWM|            : {std_pwm:.3f}\n")
    #     f.write(f"Maximum overshoot    : {max_overshoot:.6f} rad ({np.degrees(max_overshoot):.3f} deg)\n")

    # print(f"Metrics saved to '{output_file}'")

    return (stable_time, mean_alpha, std_alpha, mean_pwm, std_pwm, max_overshoot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("output_file", help ="Input name of output folder")
    args = parser.parse_args()

    # output_dir = args.output_dir

    output_file = args.output_file

    results = []

    for i in range(1, 11):
        input_file = f"{args.input_file}{i:02d}.csv"
        metrics = compute_metrics(input_file)
        results.append((i,) + metrics)

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

    print(f"Results written to {output_file}")
