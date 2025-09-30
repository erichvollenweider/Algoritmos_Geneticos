import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix, hamming_loss

# --- Cargar dataset ---
df = pd.read_csv("./Practico3/dataset_pelicula.csv")

# Features (título transformado a vectores)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Titulo"])

# Etiquetas multilabel (géneros)
y = df[["Accion", "Ciencia_Ficcion", "Romance", "Comedia", "Aventura", "Terror"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo OneVsRest para multilabel
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

print("=== Resultados One-vs-Rest (multilabel) ===")
print(classification_report(y_test, y_pred, target_names=y.columns))
print("Hamming Loss:", hamming_loss(y_test, y_pred))

# Matriz de confusión por etiqueta
cms = multilabel_confusion_matrix(y_test, y_pred)
for i, genre in enumerate(y.columns):
    print(f"\nMatriz de confusión para {genre}:")
    print(cms[i])


# --- Obtener reporte como diccionario ---
report = classification_report(y_test, y_pred, target_names=y.columns, output_dict=True)

# Extraer valores dinámicos
etiquetas = list(y.columns)
precision = [report[genre]["precision"] for genre in etiquetas]
recall = [report[genre]["recall"] for genre in etiquetas]
f1 = [report[genre]["f1-score"] for genre in etiquetas]

# --- Graficar ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(etiquetas, precision, color="skyblue")
axes[0].set_title("Precisión")
axes[0].set_ylim(0, 1)

axes[1].bar(etiquetas, recall, color="orange")
axes[1].set_title("Recall")
axes[1].set_ylim(0, 1)

axes[2].bar(etiquetas, f1, color="green")
axes[2].set_title("F1-score")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.show()