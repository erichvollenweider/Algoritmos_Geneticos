import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder

# === 1. Dataset ===
# Columnas: Parcial1, Parcial2, Asistencia
X = [
    ["BAJA", "BAJA", "NO"],
    ["BAJA", "ALTA", "NO"],
    ["MEDIA", "ALTA", "NO"],
    ["ALTA", "ALTA", "SI"],
    ["MEDIA", "BAJA", "SI"],
    ["ALTA", "ALTA", "NO"],
    ["BAJA", "BAJA", "SI"],
    ["ALTA", "BAJA", "SI"],
    ["MEDIA", "ALTA", "SI"],
    ["ALTA", "ALTA", "SI"]
]

# Etiquetas (resultado final del alumno)
y = ["REPROBADO", "REPROBADO", "APROBADO", "PROMOCION", "APROBADO",
     "APROBADO", "REPROBADO", "APROBADO", "PROMOCION", "PROMOCION"]

columnas = ["Horas", "Asistencia", "Tareas"]

# === 2. Convertir a DataFrame ===
df = pd.DataFrame(X, columns=columnas)

# === 3. Codificar variables categóricas ===
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(df).toarray()

# === 4. Entrenar el modelo ===
clf = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
clf.fit(X_encoded, y)

# === 5. Visualizar el árbol ===
plt.figure(figsize=(16, 10))
plot_tree(
    clf,
    filled=True,
    fontsize=8,
    feature_names=encoder.get_feature_names_out(columnas),
    class_names=clf.classes_
)
plt.title("Árbol de Decisión - Aprobados y Promociones")
plt.show()

# === 6. Probar con una nueva instancia ===
nueva_instancia = pd.DataFrame([["ALTA", "BAJA", "SI"]], columns=columnas)
nueva_encoded = encoder.transform(nueva_instancia).toarray()

prediccion = clf.predict(nueva_encoded)
print("El alumno está:", prediccion[0])
