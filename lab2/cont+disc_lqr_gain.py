import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.signal import cont2discrete

# 1. Define continuous model (A, B)
# Note: Verify A against your original class manual if needed.
A = np.array([
    [0, 1.0, 0, 0],
    [0, -11.3205, -49.9955, 0.4820],
    [0, 0, 0, 1.0],
    [0, 16.9206, 171.2602, -1.6510]
])

B = np.array([
    [0],
    [0.8171],
    [0],
    [-1.2213]
])

# Sample time
Ts = 0.005

# 2. Select Q and R weighting matrices
# Q prioritizes states [theta, theta_dot, alpha, alpha_dot]
Q = np.diag([1.0, 0.1, 10.0, 1.0])
# R penalizes control effort
R = np.array([[0.01]])

print("=== Continuous-Model Design ===")
# Solve Continuous Algebraic Riccati Equation (CARE)
Pc = solve_continuous_are(A, B, Q, R)
Kc = np.linalg.inv(R) @ B.T @ Pc
eig_cont, _ = np.linalg.eig(A - B @ Kc)

print("Pc:\n", np.round(Pc, 4))
print("Kc:\n", np.round(Kc, 4))
print("Closed-loop Eigenvalues (Continuous):\n", np.round(eig_cont, 4))

print("\n=== Discrete-Model Design ===")
# ZOH Discretization
sys_d = cont2discrete((A, B, np.eye(4), np.zeros((4,1))), Ts, method="zoh")
Ad, Bd = sys_d[0], sys_d[1]

# Solve Discrete Algebraic Riccati Equation (DARE)
Pd = solve_discrete_are(Ad, Bd, Q, R)
Kd = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad
eig_disc, _ = np.linalg.eig(Ad - Bd @ Kd)

print("Ad:\n", np.round(Ad, 4))
print("Bd:\n", np.round(Bd, 4))
print("Pd:\n", np.round(Pd, 4))
print("Kd:\n", np.round(Kd, 4))
print("Closed-loop Eigenvalues (Discrete):\n", np.round(eig_disc, 4))
