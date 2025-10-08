import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

# === 1. Cargar dataset ===
df = pd.read_csv("./Practico4/ScreenTime_vs_MentalWellness.csv") 
df = df.fillna(df.mean(numeric_only=True))

# === 2. Separar variables ===
X = df.drop("sleep_quality_1_5", axis=1)
y = df["sleep_quality_1_5"]

# Convertir categóricas a numéricas
X = pd.get_dummies(X, drop_first=True)

# === 3. Partición entrenamiento/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 4. Modelo base: Árbol de decisión ===
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

# === 5. Modelo con Bagging: Random Forest ===
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_samples=0.8,
    random_state=42
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# === 6. Evaluación individual ===
print("=== Árbol de Decisión ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# === 7. Matrices de confusión ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Árbol de decisión")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.show()

# === 8. Comparación de métricas ===
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average="weighted", zero_division=0)
rec_dt = recall_score(y_test, y_pred_dt, average="weighted", zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, average="weighted", zero_division=0)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)

metrics = pd.DataFrame({
    "Modelo": ["Decision Tree", "Random Forest"],
    "Accuracy": [acc_dt, acc_rf],
    "Precision": [prec_dt, prec_rf],
    "Recall": [rec_dt, rec_rf],
    "F1-score": [f1_dt, f1_rf]
})

print("\n=== Comparación de métricas ===")
print(metrics)

# === 9. Gráfico comparativo ===
plt.figure(figsize=(8, 5))
sns.barplot(x="Modelo", y="value", hue="variable",
            data=metrics.melt(id_vars="Modelo"), palette="Set2")

plt.title("Comparación de métricas entre modelos")
plt.ylabel("Valor")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# === 10. Comparación de predicciones (opcional) ===
comparison = pd.DataFrame({
    "Real": y_test.values[:20],
    "DecisionTree": y_pred_dt[:20],
    "RandomForest": y_pred_rf[:20]
})
print("\n=== Comparación de primeras predicciones ===")
print(comparison)
