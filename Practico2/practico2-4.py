import pandas as pd                                             # lectura y manipulación de tablas (DataFrame).
import numpy as np                                              # operaciones numéricas (vectores/matrices).
import matplotlib.pyplot as plt                                 # gráficos y visualización.
import seaborn as sns                                           # gráficos y visualización.
from sklearn.model_selection import train_test_split            # separar dataset en entrenamiento/validación.
from sklearn.linear_model import LinearRegression               # modelo de regresión lineal ordinaria (OLS).
from sklearn.metrics import mean_squared_error, r2_score        # métricas para evaluar regresión.
from sklearn.preprocessing import StandardScaler                # escalado (media 0, desviación 1) de features.


# Cargar dataset
df = pd.read_csv("./Practico2/winequality-red.csv", sep=";") # este dataset esta separado por ";"
print(df.head()) # muestra las primeras filas para verificar que se cargó bien.


# Explorar dataset
print(df.info())
print(df.describe())
print(df.corr()["quality"].sort_values(ascending=False))

sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.show()


# Separar variables predictoras y target
X = df.drop("quality", axis=1) # X son las variables predictoras (todas menos quality).
y = df["quality"]              # y es la variable objetivo (calidad).


# División en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
# test_size=0.3: el 30% se reserva para validación, 70% para entrenar.
# random_state=42: fija la semilla para obtener siempre la misma separación (reproducibilidad).


# Escalado (importante en regresión)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit_transform calcula media/desv en X_train y aplica la transformación.
X_val_scaled = scaler.transform(X_val)         # transform aplica la misma transformación calculada a X_val.


# Entrenar un modelo de regresión lineal
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
# Instancias el modelo y lo ajustas por mínimos cuadrados ordinarios (OLS) usando X_train_scaled y y_train.


y_pred = lin_reg.predict(X_val_scaled)              # predicciones sobre el set de validación.

print("MSE:", mean_squared_error(y_val, y_pred))    # error cuadrático medio — penaliza fuertemente errores grandes.
print("R²:", r2_score(y_val, y_pred))               # proporción de varianza de y explicada por el modelo (1 = perfecto, 0 = igual que la media, negativo = peor que la media).


# Analizar coeficientes
coef = pd.DataFrame({
    "Feature": X.columns,
    "Coeficiente": lin_reg.coef_
}).sort_values(by="Coeficiente", ascending=False)

print(coef)
