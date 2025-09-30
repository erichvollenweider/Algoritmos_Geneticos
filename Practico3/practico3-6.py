import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Cargar dataset ---
df = pd.read_csv("./Practico3/dataset_cerveza.csv")

X = df[["SRM", "IBU", "ABV"]]  # características
y = df["CLASE"]                # etiquetas numéricas

mapa_clases = {
    1: "Blanca",
    2: "Lager",
    3: "Pilsner",
    4: "IPA",
    5: "Fuerte",
    6: "Vino de cebada",
    7: "Portero",
    8: "Cerveza fuerte belga"
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Modelo base: Logistic Regression ---
base_model = LogisticRegression(max_iter=1000)

# ---------------------------
# 1) One-vs-Rest (OVR)
# ---------------------------
ovr_clf = OneVsRestClassifier(base_model)
ovr_clf.fit(X_train, y_train)
y_pred_ovr = ovr_clf.predict(X_test)

print("=== Resultados OVR (One-vs-Rest) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ovr))
print(classification_report(y_test, y_pred_ovr, target_names=[mapa_clases[c] for c in sorted(mapa_clases.keys())]))

cm_ovr = confusion_matrix(y_test, y_pred_ovr)
plt.figure(figsize=(7,5))
sns.heatmap(cm_ovr, annot=True, fmt="d", cmap="Blues",
            xticklabels=[mapa_clases[c] for c in sorted(mapa_clases.keys())],
            yticklabels=[mapa_clases[c] for c in sorted(mapa_clases.keys())])
plt.title("Matriz de confusión - OVR")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

# ---------------------------
# 2) One-vs-One (OVO)
# ---------------------------
ovo_clf = OneVsOneClassifier(base_model)
ovo_clf.fit(X_train, y_train)
y_pred_ovo = ovo_clf.predict(X_test)

print("=== Resultados OVO (One-vs-One) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ovo))
print(classification_report(y_test, y_pred_ovo, target_names=[mapa_clases[c] for c in sorted(mapa_clases.keys())]))

cm_ovo = confusion_matrix(y_test, y_pred_ovo)
plt.figure(figsize=(7,5))
sns.heatmap(cm_ovo, annot=True, fmt="d", cmap="Oranges",
            xticklabels=[mapa_clases[c] for c in sorted(mapa_clases.keys())],
            yticklabels=[mapa_clases[c] for c in sorted(mapa_clases.keys())])
plt.title("Matriz de confusión - OVO")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()


# --- Softmax ---

# X = features, y = clases (del dataset de cervezas)
X = df[["SRM", "IBU", "ABV"]]
y = df["CLASE"]

# División train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelo con softmax
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)
print("Predicciones: ", y_pred)

# Reporte de métricas
print("=== Resultados Softmax (multinomial) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mapa_clases.values()))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=mapa_clases.values(), yticklabels=mapa_clases.values())
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de confusión - Softmax (multinomial)")
plt.show()