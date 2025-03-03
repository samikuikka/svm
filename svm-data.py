import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Generate synthetic 2D data (same as before)
# -------------------------------------------------------
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

# -------------------------------------------------------
# Visualize the dataset
# -------------------------------------------------------
plt.figure(figsize=(8,6))

# Plot Class -1
plt.scatter(X1[:, 0], X1[:, 1], color='red', marker='o', label='Class -1')

# Plot Class +1
plt.scatter(X2[:, 0], X2[:, 1], color='blue', marker='s', label='Class +1')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Synthetic Dataset for SVM")
plt.legend()
plt.grid(True)
plt.show()
