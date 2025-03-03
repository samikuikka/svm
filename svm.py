import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

np.random.seed(42)

n_samples = 50

# Class -1
X1 = np.random.randn(n_samples, 2) * 0.8 + np.array([-2, -2])
y1 = -1 * np.ones(n_samples)

# Class +1
X2 = np.random.randn(n_samples, 2) * 0.8 + np.array([2, 2])
y2 = np.ones(n_samples)

# Combine
X = np.vstack([X1, X2])
y = np.hstack([y1, y2])

def svm_solver(X, y, C=1.0):

    N = X.shape[0]
    
    # Convert y to a column vector
    y = y.astype(float).reshape(-1, 1)

    # K[i,j] = X[i] dot X[j]
    K = np.dot(X, X.T)
    # P = outer product of y with itself times the Gram matrix
    P = matrix(np.outer(y, y) * K, tc='d')

    # q is a vector of -1's (because we are minimizing -sum(α_i))
    q = matrix(-np.ones((N, 1)), tc='d')

    # -----------------------------
    # Inequality constraints: 0 ≤ α_i ≤ C
    G_std = -np.eye(N) # -α_i ≤ 0  => α_i ≥ 0
    G_slack = np.eye(N) # α_i ≤ C
    G = matrix(np.vstack((G_std, G_slack)), tc='d')

    h_std = np.zeros(N)
    h_slack = np.ones(N) * C
    h = matrix(np.hstack((h_std, h_slack)), tc='d')

    # -----------------------------
    # Equality constraint: sum(α_i y_i) = 0
    A = matrix(y.reshape(1, -1), tc='d')
    b = matrix(np.zeros(1), tc='d')

    #Solves a quadratic program
    #
    # minimize    (1/2)*x'*P*x + q'*x
    # subject to  G*x <= h
    #            A*x = b.
    solvers.options['show_progress'] = False  # suppress solver output
    solution = solvers.qp(P, q, G, h, A, b)
    
    alpha = np.ravel(solution['x'])
    return alpha

C = 1.0
alpha = svm_solver(X, y, C=1.0)


def compute_w_b(alpha, X, y, C=1.0, threshold=1e-5):
    w = np.sum((alpha * y.reshape(-1))[:, None] * X, axis=0)
    
    support_indices = np.where((alpha > threshold) & (alpha < C - threshold))[0]
    if len(support_indices) == 0:
        support_indices = np.where(alpha > threshold)[0]
    
    # Compute bias b using support vectors:
    # For any support vector: y_i = sign(w^T x_i + b) and ideally y_i (w^T x_i + b) = 1.
    b_values = [y[i] - np.dot(w, X[i]) for i in support_indices]
    b = np.mean(b_values)
    return w, b, support_indices

w, b, support_indices = compute_w_b(alpha, X, y, C=C)
print("Weight vector w:", w)
print("Bias b:", b)

plt.figure(figsize=(8, 6))

# Separate classes for visualization
X_neg = X[y == -1]
X_pos = X[y == 1]

plt.scatter(X_neg[:, 0], X_neg[:, 1], color='red', marker='o', label='Class -1')
plt.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', marker='s', label='Class +1')

# Highlight support vectors
plt.scatter(X[support_indices, 0], X[support_indices, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')

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
plt.title("Soft Margin SVM: Data, Support Vectors, and Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
