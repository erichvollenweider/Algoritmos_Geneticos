import numpy as np


def step(z):
    return 1 if z >= 0 else 0


# Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([0,1,1,1]) # Clase

# Hiperparametros
eta = 1 # Learning Rate
epochs = 4 # Cantidad de epocas para el entrenamiento
w = np.array([0,0]) # Pesos iniciales
b = 0

# Entrenamiento
for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        z = np.dot(w, xi) + b
        y_pred = step(z)
        error = target - y_pred
        if error != 0:
            # Actualizacion
            w = w + eta * error * xi
            b = b + eta * error
            errors += 1

    print(f"Época {epoch}, errores: {errors}, w={w}, b={b}")
    if errors == 0:
        print("Convergió en época", epoch)
        break
