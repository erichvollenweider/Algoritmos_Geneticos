import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import DecisionBoundaryDisplay



# Dataset make_moons
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

# Distintos kernels
kernels = ["linear", "poly", "rbf", "sigmoid"]
C_values = [0.1, 1, 10]

fig, axes = plt.subplots(len(kernels), len(C_values), figsize=(15, 12))


for i, kernel in enumerate(kernels):
    for j, C in enumerate(C_values):
        svm = SVC(kernel=kernel, C=C)
        svm.fit(X, y)

        disp = DecisionBoundaryDisplay.from_estimator(
            svm, X, response_method="predict",
            alpha=0.6, cmap="coolwarm", ax=axes[i, j]
        )

        axes[i, j].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k")
        axes[i, j].set_title(f"Kernel={kernel}, C={C}")

plt.tight_layout()
plt.show()
