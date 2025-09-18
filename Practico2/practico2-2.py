import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fijamos la semilla para reproducibilidad
np.random.seed(42)

# Generamos valores de x
x = np.linspace(0, 10, 50)  # 50 puntos entre 0 y 10

# Generamos ruido e ~ N(0,1)
e = np.random.normal(0, 1, size=x.shape)

# Definimos la función con ruido
y = 3*x + 2 + e

# Mostrar algunos valores
print("Ejemplo de 5 datos (x, y):")
for i in range(5):
    print(f"x={x[i]:.2f}, y={y[i]:.2f}")

# Graficamos
plt.scatter(x, y, color="blue", label="Datos con ruido")
plt.plot(x, 3*x + 2, color="red", label="Recta ideal: y=3x+2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


def ecu_normal_y_scikit_learn():
    m = 100
    X = np.random.randn(m, 1) # Atributos
    y = 3*X + 2 + np.random.randn(m, 1)

    X_Ones = np.c_[np.ones((100, 1)), X]

    theta_normal_ec = np.linalg.inv(X_Ones.T.dot(X_Ones)).dot(X_Ones.T).dot(y)
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    return {
        'ecuacion_normal': [theta_normal_ec[0][0], theta_normal_ec[1][0]],
        'scikit_learn': [lin_reg.intercept_[0], lin_reg.coef_[0][0]]
    }

print(ecu_normal_y_scikit_learn())

#--------------------------------------(b)----------------------------------------#

# Inicializamos parámetros
w = 0.0
b = 0.0
alpha = 0.01  # learning rate
epochs = 1000
m = len(x)

# Gradiente Descente
for _ in range(epochs):
    y_pred = w*x + b
    error = y - y_pred

    dw = -(2/m) * np.sum(x * error)
    db = -(2/m) * np.sum(error)

    w -= alpha * dw
    b -= alpha * db

print(f"w aprendido = {w:.4f}")
print(f"b aprendido = {b:.4f}")

# Graficar resultados
plt.scatter(x, y, color="blue", label="Datos con ruido")
plt.plot(x, 3*x + 2, color="red", label="Recta real y=3x+2")
plt.plot(x, w*x + b, color="green", label=f"Recta aprendida y={w:.2f}x+{b:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

