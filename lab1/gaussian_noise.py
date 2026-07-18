import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from scipy import stats

# To run this script, in the terminal run the following command in this folder: 
# $ python calculate_noise.py [simulation_file] [experiment_file]
# The script will automatically create a txt file with the noise parameters

def compute_noise_params(sim_file, exp_file):
    """
    Calculate Gaussian noise parameters by comparing simulation and experimental data.
    Noise is modeled as: noise = experiment - simulation ~ N(mu, sigma^2)
    """
    
    # Read CSVs
    sim_df = pd.read_csv(sim_file)
    exp_df = pd.read_csv(exp_file)
    
    # ---------- Column names ----------
    TIME_COL = "time_s"
    THETA_COL = "theta_rad"
    ALPHA_COL = "alpha_rad"
    PWM_COL = "pwm"
    
    # Check required columns
    required_sim = [TIME_COL, THETA_COL, ALPHA_COL, PWM_COL]
    required_exp = [TIME_COL, THETA_COL, ALPHA_COL, PWM_COL]
    
    missing_sim = [c for c in required_sim if c not in sim_df.columns]
    missing_exp = [c for c in required_exp if c not in exp_df.columns]
    
    if missing_sim:
        raise ValueError(f"Missing columns in simulation file: {missing_sim}")
    if missing_exp:
        raise ValueError(f"Missing columns in experiment file: {missing_exp}")
    
    # Merge on time_s
    merged = pd.merge(sim_df, exp_df, on=TIME_COL, suffixes=('_sim', '_exp'))
    
    # Compute differences (experiment - simulation)
    merged['theta_diff'] = merged[f'{THETA_COL}_exp'] - merged[f'{THETA_COL}_sim']
    merged['alpha_diff'] = merged[f'{ALPHA_COL}_exp'] - merged[f'{ALPHA_COL}_sim']
    merged['pwm_diff'] = merged[f'{PWM_COL}_exp'] - merged[f'{PWM_COL}_sim']
    
    def get_noise_params(diff_series):
        """Calculate mean and standard deviation with outlier removal."""
        # Remove NaN values
        clean_data = diff_series.dropna()
        
        if len(clean_data) < 2:
            return 0.0, 0.0
        
        # Remove outliers beyond 3 sigma for robust estimation
        z_scores = np.abs(stats.zscore(clean_data))
        filtered = clean_data[z_scores < 3]
        
        if len(filtered) < 2:
            return float(np.mean(clean_data)), float(np.std(clean_data, ddof=1))
        
        mu = float(np.mean(filtered))
        sigma = float(np.std(filtered, ddof=1))
        return mu, sigma
    
    # Calculate parameters for each variable
    theta_mu, theta_sigma = get_noise_params(merged['theta_diff'])
    alpha_mu, alpha_sigma = get_noise_params(merged['alpha_diff'])
    pwm_mu, pwm_sigma = get_noise_params(merged['pwm_diff'])
    
    return (theta_mu, theta_sigma, alpha_mu, alpha_sigma, pwm_mu, pwm_sigma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_file", help="Simulation CSV file (noise-free)")
    parser.add_argument("exp_file", help="Experimental CSV file (with noise)")
    args = parser.parse_args()
    
    # Compute noise parameters
    (theta_mu, theta_sigma, alpha_mu, alpha_sigma, pwm_mu, pwm_sigma) = compute_noise_params(args.sim_file, args.exp_file)
    
    # Create output filename from simulation filename
    sim_path = Path(args.sim_file)
    output_file = sim_path.stem + "_noise_parameters.txt"
    
    # Write results to output file
    with open(output_file, "w", encoding='utf-8') as f:
        f.write("Noise Parameter | mu (mean)     | sigma (std dev) | Unit\n")
        f.write("-" * 70 + "\n")
        f.write(f"theta           | {theta_mu:>12.6f} | {theta_sigma:>12.6f} | rad\n")
        f.write(f"alpha           | {alpha_mu:>12.6f} | {alpha_sigma:>12.6f} | rad\n")
        f.write(f"PWM             | {pwm_mu:>12.6f} | {pwm_sigma:>12.6f} | PWM\n")
    
    print(f"Noise parameters written to {output_file}")