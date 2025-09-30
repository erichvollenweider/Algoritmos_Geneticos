import numpy as np

# Dataset de la tabla
X = np.array([[0,0],
              [0,1],
              [1,0],
              [2,2],
              [2,3],
              [3,2]])

y = np.array([0,0,0,1,1,1])

# Parámetros
eta = 1       # tasa de aprendizaje
epochs = 20   # número máximo de épocas

# Inicialización de pesos y sesgo
w = np.zeros(X.shape[1])
b = 0

# Función escalón
def step(z):
    return 1 if z >= 0 else 0

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
            # Actualización de pesos
            w = w + eta * error * xi
            b = b + eta * error
            errors += 1
    print(f"Época {epoch}, errores: {errors}, w={w}, b={b}")
    if errors == 0:
        print("✅ Convergió en época", epoch)
        break


# Dataset de la tabla
X = np.array([[0,0],
              [0,1],
              [1,0],
              [2,2],
              [2,3],
              [3,2]])

y = np.array([0,0,0,1,1,1])

# Inicialización
m, n = X.shape
w = np.zeros(n)
b = 0
eta = 0.1
epochs = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Entrenamiento
for epoch in range(epochs):
    Z = np.dot(X, w) + b
    A = sigmoid(Z)
    cost = -(1/m) * np.sum(y*np.log(A+1e-9) + (1-y)*np.log(1-A+1e-9))
    
    # Gradientes
    dw = (1/m) * np.dot(X.T, (A - y))
    db = (1/m) * np.sum(A - y)
    
    # Actualización
    w -= eta * dw
    b -= eta * db

    if epoch % 100 == 0:
        print(f"Época {epoch}, costo={cost:.4f}, w={w}, b={b}")

print("Pesos finales:", w, "b=", b)

# Predicciones
y_pred = (sigmoid(np.dot(X, w) + b) >= 0.5).astype(int)
print("Predicciones:", y_pred)