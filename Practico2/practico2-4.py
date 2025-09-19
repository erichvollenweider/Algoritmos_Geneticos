import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Cargar dataset
df = pd.read_csv("./Practico2/winequality-red.csv", sep=";") # este dataset esta separado por ";"
 
print(df.head())

# Explorar dataset
print(df.info())
print(df.describe())
print(df.corr()["quality"].sort_values(ascending=False))

sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.show()

# Separar variables predictoras y target
X = df.drop("quality", axis=1)
y = df["quality"]

# División en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalado (importante en regresión)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Entrenar un modelo de regresión lineal
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

y_pred = lin_reg.predict(X_val_scaled)

print("MSE:", mean_squared_error(y_val, y_pred))
print("R²:", r2_score(y_val, y_pred))

# Analizar coeficientes
coef = pd.DataFrame({
    "Feature": X.columns,
    "Coeficiente": lin_reg.coef_
}).sort_values(by="Coeficiente", ascending=False)

print(coef)
