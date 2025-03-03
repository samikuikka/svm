import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# -------------------------------------------------------
# Generate synthetic 2D data with more overlap
# -------------------------------------------------------
np.random.seed(42)
n_samples = 50

# Class -1
X1 = np.random.randn(n_samples, 2) * 1.0 + np.array([-1, -1])
y1 = -1 * np.ones(n_samples)

# Class +1 
X2 = np.random.randn(n_samples, 2) * 1.0 + np.array([1, 1])
y2 = np.ones(n_samples)

# Combine into one dataset
X = np.vstack([X1, X2])
y = np.hstack([y1, y2])

# -------------------------------------------------------
# SVM Solver using cvxopt to compute alpha values
# -------------------------------------------------------
def svm_solver(X, y, C=1.0):
    N = X.shape[0]
    
    # Convert y to a column vector
    y = y.astype(float).reshape(-1, 1)
    
    # Compute the Gram matrix: K[i,j] = X[i] dot X[j]
    K = np.dot(X, X.T)
    
    # P[i,j] = y_i y_j (X[i] dot X[j])
    P = matrix(np.outer(y, y) * K, tc='d')
    
    # q is a vector of -1's (because we are minimizing -sum(α))
    q = matrix(-np.ones((N, 1)), tc='d')
    
    # Inequality constraints: 0 ≤ α_i ≤ C
    G_std = -np.eye(N)  # -α_i ≤ 0  =>  α_i ≥ 0
    G_slack = np.eye(N) # α_i ≤ C
    G = matrix(np.vstack((G_std, G_slack)), tc='d')
    
    h_std = np.zeros(N)
    h_slack = np.ones(N) * C
    h = matrix(np.hstack((h_std, h_slack)), tc='d')
    
    # Equality constraint: sum(α_i y_i) = 0
    A = matrix(y.reshape(1, -1), tc='d')
    b = matrix(np.zeros(1), tc='d')
    
    # Solve QP problem using cvxopt
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    
    alpha = np.ravel(solution['x'])
    return alpha

# Solve for α
C = 1.0
alpha = svm_solver(X, y, C=C)
print("Optimal alpha values:")
print(alpha)

# -------------------------------------------------------
# Compute weight vector w and bias b from optimal alpha
# -------------------------------------------------------
def compute_w_b(alpha, X, y, C=1.0, threshold=1e-5):
    w = np.sum((alpha * y.reshape(-1))[:, None] * X, axis=0)
    support_indices = np.where((alpha > threshold) & (alpha < C - threshold))[0]
    if len(support_indices) == 0:
        support_indices = np.where(alpha > threshold)[0]
    b_values = [y[i] - np.dot(w, X[i]) for i in support_indices]
    b = np.mean(b_values)
    return w, b, support_indices

w, b, support_indices = compute_w_b(alpha, X, y, C=C)
print("Weight vector w:", w)
print("Bias b:", b)

# -------------------------------------------------------
# Visualization: Plot data, support vectors, and decision boundary
# -------------------------------------------------------
plt.figure(figsize=(8, 6))

# Separate classes for visualization
X_neg = X[y == -1]
X_pos = X[y == 1]

plt.scatter(X_neg[:, 0], X_neg[:, 1], color='red', marker='o', label='Class -1')
plt.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', marker='s', label='Class +1')

# Highlight support vectors
plt.scatter(X[support_indices, 0], X[support_indices, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')

# Create mesh grid to plot decision boundary and margins
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Decision function: f(x) = w^T x + b
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
Z = Z.reshape(xx.shape)

# Plot decision boundary (f(x)=0) and margins (f(x)=1 and f(x)=-1)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Soft Margin SVM with Overlapping Data")
plt.legend()
plt.grid(True)
plt.show()
