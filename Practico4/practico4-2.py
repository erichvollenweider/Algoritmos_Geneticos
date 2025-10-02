import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

# 1) Generar dataset sintético
X, y = make_moons(n_samples=100, noise=0.5, random_state=42)

# 2) Definir valores de k a probar
k_values = [1, 15, 50]

# 3) Crear una figura con subplots
fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(16, 4))


# -------------------
# (a) Graficar dataset original
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
axes[0].set_title("Dataset - make_moons")
axes[0].set_xlabel("X1")
axes[0].set_ylabel("X2")

# -------------------
# (b) Entrenar y graficar decisión para cada k
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
)

for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree")
    knn.fit(X, y)

    # Frontera de decisión
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[i+1].contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.3)
    axes[i+1].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")

    axes[i+1].set_title(f"KNN (k={k})")
    axes[i+1].set_xlabel("X1")
    axes[i+1].set_ylabel("X2")

plt.tight_layout()
plt.show()


#   Conclusión:
#       KNN con k=1:
#           La frontera es muy irregular.
#           El modelo “memoriza” casi cada punto de entrenamiento → overfitting.
#           Se adapta incluso al ruido (zonas raras con un punto solitario cambian la frontera).

#       KNN con k=5:
#           La frontera se suaviza.
#           El modelo toma en cuenta más vecinos para decidir, por lo que generaliza mejor.

#       KNN con k=15:
#           frontera se vuelve muy suave.
#           El modelo pierde detalles importantes → underfitting.
#           Mezcla demasiado las clases, a veces clasificando mal puntos que están claramente dentro de un grupo.