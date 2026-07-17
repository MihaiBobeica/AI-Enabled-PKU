import numpy as np

try:
    from scipy.linalg import solve_continuous_are
except Exception as e:
    print("[ERR] scipy is required. Install with: pip install scipy numpy")
    raise

# Linearized RIP model around upright equilibrium.
# State: x = [theta, theta_dot, alpha, alpha_dot]^T
A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -11.3205, -49.9955, 0.4820],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 16.9206, 171.2602, -1.6510],
])
B = np.array([
    [0.0],
    [0.8171],
    [0.0],
    [-1.2213],
])

# Students should tune Q and R.
Q = np.diag([1.0, 0.1, 100.0, 1.0])
R = np.array([[1.0]])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

np.set_printoptions(precision=6, suppress=True)
print("P =")
print(P)
print("\nK = R^{-1} B^T P =")
print(K)
print("\nPaste into rip_lqr_control.ino:")
print(f"static const float K_THETA     = {K[0,0]:.6f}f;")
print(f"static const float K_THETA_DOT = {K[0,1]:.6f}f;")
print(f"static const float K_ALPHA     = {K[0,2]:.6f}f;")
print(f"static const float K_ALPHA_DOT = {K[0,3]:.6f}f;")
