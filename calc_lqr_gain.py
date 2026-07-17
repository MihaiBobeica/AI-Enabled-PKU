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

#result with given parameter:
#P =
#[[  14.197279    4.759066   41.047197    4.002811]
#[   4.759066   36.849886  578.371999   47.62325 ]
#[  41.047197  578.371999 9179.597319  753.441465]
#[   4.002811   47.62325   753.441465   61.938413]]

#K = R^{-1} B^T P =
#[[  -1.        -28.052233 -447.590301  -36.732426]]

#Paste into rip_lqr_control.ino:
#static const float K_THETA     = -1.000000f;
#static const float K_THETA_DOT = -28.052233f;
#static const float K_ALPHA     = -447.590301f;
#static const float K_ALPHA_DOT = -36.732426f;
