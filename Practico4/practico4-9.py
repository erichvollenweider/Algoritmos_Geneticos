import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1. Cargar dataset Iris ===
iris = load_iris()
X = iris.data
y = iris.target

# === 2. Separar entrenamiento y test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 3. Modelos base ===
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('lr', LogisticRegression(max_iter=1000))
]

# === 4. Meta-modelos a probar ===
meta_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42)
}

# === 5. Entrenar y evaluar cada stacking ===
results = []

for name, meta in meta_models.items():
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        passthrough=False,  # si True, incluye X original junto con las predicciones base
        cv=5,               # validación cruzada para entrenar el meta-modelo
        n_jobs=-1
    )

    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))

    print(f"\n=== Meta-modelo: {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f"Matriz de confusión - Meta-modelo: {name}")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.show()

# === 6. Comparación gráfica de accuracy ===
df_results = pd.DataFrame(results, columns=["Meta-modelo", "Accuracy"])
plt.figure(figsize=(7,5))
sns.barplot(x="Meta-modelo", y="Accuracy", data=df_results, palette="viridis")
plt.title("Comparación de modelos Stacking según meta-modelo")
plt.ylim(0.8, 1.0)
plt.show()
