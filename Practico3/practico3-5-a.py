import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

np.random.seed(42)

# -------------------------
# 1) Generar dataset sintético
# -------------------------
n_per_class = 100

# No-Stout: IBU y RMS centrados en valores bajos/medios
ibu_nostout = np.random.normal(loc=15, scale=5, size=n_per_class)
rms_nostout = np.random.normal(loc=20, scale=8, size=n_per_class)

# Stout: IBU y RMS más altos
ibu_stout = np.random.normal(loc=45, scale=6, size=n_per_class)
rms_stout = np.random.normal(loc=60, scale=10, size=n_per_class)

X_nostout = np.column_stack([ibu_nostout, rms_nostout])
X_stout = np.column_stack([ibu_stout, rms_stout])

X = np.vstack([X_nostout, X_stout])
y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])  # 0 = No-Stout, 1 = Stout

# Mezclar los datos
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# -------------------------
# 2) Train / Validation split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# -------------------------
# 3) Escalado (muy recomendable)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# -------------------------
# 4) Regresión logística manual (vectorizada)
# -------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic(X, y, lr=0.1, n_iter=1000, clip_eps=1e-9):
    m, n = X.shape
    #  theta: (n+1, 1) including bias
    X_b = np.hstack([np.ones((m,1)), X])       # (m, n+1)
    theta = np.zeros((n+1, 1))                 # column vector
    y_col = y.reshape(-1,1)

    costs = []
    for it in range(n_iter):
        Z = X_b @ theta                        # (m,1)
        A = sigmoid(Z)
        # evitar log(0)
        A_clipped = np.clip(A, clip_eps, 1-clip_eps)

        cost = - (1/m) * np.sum(y_col * np.log(A_clipped) + (1 - y_col) * np.log(1 - A_clipped))
        costs.append(cost)

        # gradientes (vectorizados)
        grad = (1/m) * (X_b.T @ (A - y_col))   # shape (n+1, 1)

        # actualización
        theta = theta - lr * grad

        if it % (n_iter // 10) == 0:
            print(f"Iter {it:4d}  cost={cost:.6f}")
    return theta, costs

# Entrenar
theta, costs = train_logistic(X_train_scaled, y_train, lr=0.5, n_iter=2000)  # lr ajustable

# -------------------------
# 5) Evaluación en validación
# -------------------------
def predict_proba(X, theta):
    m = X.shape[0]
    X_b = np.hstack([np.ones((m,1)), X])
    return sigmoid(X_b @ theta).ravel()

def predict(X, theta, threshold=0.5):
    return (predict_proba(X, theta) >= threshold).astype(int)

y_val_pred = predict(X_val_scaled, theta)

print("\nMétricas en validación:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred))
print("Recall:", recall_score(y_val, y_val_pred))
print("F1:", f1_score(y_val, y_val_pred))
print("Confusion matrix:\n", confusion_matrix(y_val, y_val_pred))

# -------------------------
# 6) Visualizaciones (opcional, ver frontera de decisión)
# -------------------------
# Mostrar datos y frontera
plt.figure(figsize=(7,6))
plt.scatter(X_val_scaled[y_val==0,0], X_val_scaled[y_val==0,1], c='C0', marker='o', label='No-Stout (0)')
plt.scatter(X_val_scaled[y_val==1,0], X_val_scaled[y_val==1,1], c='C1', marker='x', label='Stout (1)')

# lineal decision boundary: theta0 + theta1*x1 + theta2*x2 = 0  -> x2 = (-theta0 - theta1*x1)/theta2
theta_flat = theta.ravel()
x_vals = np.linspace(X_val_scaled[:,0].min()-0.5, X_val_scaled[:,0].max()+0.5, 100)
if abs(theta_flat[2]) > 1e-6:
    y_vals = -(theta_flat[0] + theta_flat[1]*x_vals) / theta_flat[2]
    plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Decision boundary')
plt.xlabel('IBU (estandarizado)')
plt.ylabel('RMS (estandarizado)')
plt.legend()
plt.title('Datos de validación y frontera de decisión')
plt.grid(True)
plt.show()
