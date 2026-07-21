#!/usr/bin/env python3

import argparse
import numpy as np

# Small value to avoid division by zero
EPSILON = 1e-9

# Metric names in order of the columns
METRICS = [
    "Stable time (s)",
    "Mean |alpha| (rad)",
    "Std |alpha| (rad)",
    "Mean |PWM|",
    "Std |PWM|",
    "Max overshoot (rad)"
]


def read_results(filename):
    """
    Reads a file with rows like

    1 & 1.3900 & 0.043476 & ... & 0.245676\\

    Returns an (N,6) numpy array containing only the metrics.
    """
    data = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Remove LaTeX row ending
            line = line.replace("\\\\", "")

            parts = [p.strip() for p in line.split("&")]

            if len(parts) < 7:
                continue

            # Ignore trial number
            values = list(map(float, parts[1:7]))
            data.append(values)

    return np.asarray(data)


def sim_to_real(real, sim):
    real_mean = np.mean(real, axis=0)
    sim_mean = np.mean(sim, axis=0)

    delta = np.abs(real_mean - sim_mean)
    gap = delta / np.maximum(np.abs(real_mean), EPSILON) * 100

    return real_mean, sim_mean, delta, gap


def main():
    parser = argparse.ArgumentParser(
        description="Compute Sim-to-Real analysis."
    )
    parser.add_argument("real_file", help="Physical experiment results")
    parser.add_argument("sim_file", help="Simulation experiment results")

    args = parser.parse_args()

    real = read_results(args.real_file)
    sim = read_results(args.sim_file)

    if real.shape != sim.shape:
        print("Warning: datasets have different sizes.")

    real_mean, sim_mean, delta, gap = sim_to_real(real, sim)

    output_file = "sim_to_real_configured_results.txt"

    row_names = [
    "Stable time $t_s$ / s",
    "Mean $|\\alpha|$ / rad",
    "Std. $|\\alpha|$ / rad",
    "Mean $|\\mathrm{PWM}|$",
    "Std. $|\\mathrm{PWM}|$",
    "Maximum stable-stage overshoot / rad"
    ]

    with open(output_file, "w") as f:
        for i, name in enumerate(row_names):
            f.write(
                f"{name}\n"
                f"& {real_mean[i]:.6f}"
                f" & {sim_mean[i]:.6f}"
                f" & {delta[i]:.6f}"
                f" & {gap[i]:.2f}"
                r"\\"
                "\n\n"
            )

    print(f"LaTeX table written to '{output_file}'")


if __name__ == "__main__":
    main()