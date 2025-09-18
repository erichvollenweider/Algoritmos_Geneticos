import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------

# (x, y) = {(1, 2), (2, 3), (3, 5), (4, 7), (5, 8)}

# Ajustar una recta y = wx + b usando mínimos cuadrados, calculando los coeficientes w y

# Xmed = 3
# Ymed = 5

#       ∑(xi-Xmed)(yi-Ymed)      (1-3)(2-5)+(2-3)(3-5)+(3-3)(5-5)+(4-3)(7-5)+(5-3)(8-5)
# w = ----------------------- = -------------------------------------------------------- = 1.6
#           ∑(xi-Xmed)²               (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²


# b = Ymed - w * Xmed = 5 - 1.6 * 3 = 0.2

# y = 1.6x + 0.2

# -------------------------------------------------------------------------------------------------

# Datos
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 8])

# Calcular medias
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calcular w y b (mínimos cuadrados)
w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
b = y_mean - w * x_mean

print(f"w (pendiente) = {w:.2f}")
print(f"b (intersección) = {b:.2f}")

# Gráfico
plt.scatter(x, y, color="blue", label="Datos")
plt.plot(x, w*x + b, color="red", label=f"Recta: y={w:.2f}x+{b:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
