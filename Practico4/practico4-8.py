import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# === 1. Cargar dataset ===
df = pd.read_csv("./Practico4/ScreenTime_vs_MentalWellness.csv") 
df = df.fillna(df.mean(numeric_only=True))

# === 2. Separar variables ===
X = df.drop("sleep_quality_1_5", axis=1)
y = df["sleep_quality_1_5"]

# Convertir categóricas a numéricas
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(0)  # ✅ importante: eliminar cualquier NaN restante

# === 3. Dividir en entrenamiento y prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# === 4. AdaBoost con árbol de decisión ===
base_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_clf = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)

# === 5. Gradient Boosting ===
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)

# === 6. KNN (sin boosting) ===
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)

# === 7. Evaluación ===
print("=== AdaBoost (Decision Tree base) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))

print("=== Gradient Boosting ===")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

print("=== KNN ===")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# === 8. Comparación gráfica ===
results = pd.DataFrame({
    "Modelo": ["AdaBoost (Tree)", "Gradient Boosting", "KNN"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_ada),
        accuracy_score(y_test, y_pred_gb),
        accuracy_score(y_test, y_pred_knn)
    ]
})

plt.figure(figsize=(7,5))
sns.barplot(x="Modelo", y="Accuracy", data=results, palette="viridis")
plt.title("Comparación de Accuracy entre modelos con Boosting")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
