import numpy as np
from scipy.linalg import cholesky, solve, lstsq

# GLS problem: minimize (Ax - b)^T Ω^{-1} (Ax - b)
# Step 0: Define A, b, and Ω
A = np.random.randn(100, 10)
x_true = np.random.randn(10)
p = np.diag(np.linspace(0.5, 2.0, 100))  # positive definite covariance
b = A @ x_true + np.random.multivariate_normal(np.zeros(100), p)

# Step 1: Compute Cholesky of Ω (Ω = L L^T)
L = cholesky(p, lower=True)

# Step 2: Transform to standard least squares
#A_tilde = solve(L, A)  # L^{-1} A
#b_tilde = solve(L, b)  # L^{-1} b

# Step 3: Solve OLS problem
#x_gls, residuals, rank, s = lstsq(A_tilde, b_tilde)

#print("GLS solution:", x_gls)
