import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# 1. Generar datos (X, y) con forma cuadrática
np.random.seed(42)
X = np.linspace(-3, 3, 20).reshape(-1, 1)   # 20 valores entre -3 y 3
y = X**2 + np.random.randn(20, 1) * 2       # y = x^2 + ruido

# 2. Ajustar regresión lineal
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# 3. Calcular error MSE
mse_lin = mean_squared_error(y, y_pred_lin)
print(f"MSE de la regresión lineal: {mse_lin:.2f}")

# 4. Graficar
plt.scatter(X, y, color="blue", label="Datos (y=x^2 + ruido)")
plt.plot(X, y_pred_lin, color="red", label="Recta ajustada")
plt.legend()
plt.show()

#---------------------------(b)---------------------------#

# -------- Transformar para regresión polinómica (grado 2) --------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)  # shape (20, 2) -> columnas: [x, x^2]
print("Nombres de features:", poly.get_feature_names_out())
print("Forma X_poly:", X_poly.shape)
print("Primeras filas X_poly:\n", X_poly[:5])

# -------- Ajustar regresión sobre X_poly --------
poly_reg = LinearRegression()   # regresión lineal sobre las features polinómicas
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)

print(f"MSE lineal: {mse_lin:.3f}")
print(f"MSE polinómico (grado 2): {mse_poly:.3f}")
print("Coeficientes polinomio (intercept, coef_x, coef_x2):", poly_reg.intercept_, poly_reg.coef_)

# -------- Graficar (suavizado para la curva polinómica) --------
X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot_poly = poly_reg.predict(X_plot_poly)
y_plot_lin = lin_reg.predict(X_plot)

plt.scatter(X, y, label="Datos (x^2 + ruido)")
plt.plot(X_plot, y_plot_poly, label="Polinomial grado 2", linewidth=2)
plt.plot(X_plot, y_plot_lin, label="Recta (lineal)", linestyle="--")
plt.legend()
plt.show()

#---------------------------(c)---------------------------#

X = np.linspace(-3, 3, 200).reshape(-1, 1)   # 200 valores entre -3 y 3
y = X**2 + np.random.randn(200, 1) * 2       # y = x^2 + ruido

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

train_errors = []
val_errors = []

for d in [1, 2, 5, 10]:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)
    
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))

    print(f"Grado {d}: MSE train = {train_errors[-1]:.3f}, MSE val = {val_errors[-1]:.3f}")


degrees = [1, 2, 5, 10]
for d, train_err, val_err in zip(degrees, train_errors, val_errors):
    print(f"Grado {d}: Train MSE={train_err:.3f}, Val MSE={val_err:.3f}")

plt.plot(degrees, train_errors, label="Error entrenamiento", marker="o")
plt.plot(degrees, val_errors, label="Error validación", marker="o")
plt.xlabel("Grado del polinomio")
plt.ylabel("MSE")
plt.legend()
plt.show()