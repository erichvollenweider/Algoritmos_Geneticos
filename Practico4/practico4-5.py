import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# === 1. Cargar dataset ===
df = pd.read_csv("./Practico4/frutas.csv")

# === 2. Separar características (X) y etiqueta (y) ===
X = df.drop("Fruta", axis=1)
y = df["Fruta"]

# === 3. Codificar variables categóricas ===
X_encoded = pd.get_dummies(X, drop_first=True)

# === 4. Árbol normal (sin poda) ===
tree_normal = DecisionTreeClassifier(random_state=42)
tree_normal.fit(X_encoded, y)

# === 5. Árbol con pre-pruning (poda temprana) ===
tree_pre = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,          # limita la profundidad del árbol
    min_samples_split=2,  # mínimo de muestras para dividir
    min_samples_leaf=1,   # mínimo de muestras por hoja
    random_state=42
)
tree_pre.fit(X_encoded, y)

# === 6. Árbol con post-pruning (poda posterior) ===
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_encoded, y)

# Obtenemos valores de complejidad (alpha)
path = tree_full.cost_complexity_pruning_path(X_encoded, y)
ccp_alphas = path.ccp_alphas

# Elegimos un valor intermedio de alpha (ni muy chico ni muy grande)
alpha_optimo = ccp_alphas[len(ccp_alphas)//2]

tree_post = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha_optimo)
tree_post.fit(X_encoded, y)

# === 7. Visualización ===
plt.figure(figsize=(30, 10))

plt.subplot(1, 3, 1)
plot_tree(
    tree_normal, 
    filled=True, 
    fontsize=8, 
    feature_names=X_encoded.columns,
    class_names=tree_normal.classes_
)
plt.title("Árbol sin poda")

plt.subplot(1, 3, 2)
plot_tree(
    tree_pre, 
    filled=True, 
    fontsize=8, 
    feature_names=X_encoded.columns,
    class_names=tree_pre.classes_
)
plt.title("Pre-pruning (poda temprana)")

plt.subplot(1, 3, 3)
plot_tree(
    tree_post, 
    filled=True, 
    fontsize=8, 
    feature_names=X_encoded.columns,
    class_names=tree_post.classes_
)
plt.title("Post-pruning (poda posterior)")

plt.tight_layout()
plt.show()


# === 8. Predicción para una nueva fruta ===
nueva_fruta = pd.DataFrame({
    "Color": ["Rojo"],
    "Tamaño": ["Pequeño"],
    "Peso": [160],
    "Sabor": ["Dulce"]
})

# Aplicamos el mismo one-hot encoding que se usó en el entrenamiento
nueva_fruta_encoded = pd.get_dummies(nueva_fruta, drop_first=True)

# Alineamos las columnas con las del modelo entrenado
nueva_fruta_encoded = nueva_fruta_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Elegimos el modelo (por ejemplo el árbol sin poda)
prediccion = tree_normal.predict(nueva_fruta_encoded)

print("La fruta predicha es:", prediccion[0])
