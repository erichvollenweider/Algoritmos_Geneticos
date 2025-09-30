import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Tus datos originales
X = np.array([
    [15, 20],
    [12, 15],
    [28, 39],
    [21, 30],
    [45, 20],
    [40, 61],
    [42, 70]
])
y = np.array([0, 0, 0, 0, 1, 1, 1])

# Estandarizar (MUY importante)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic(X, y, lr=0.05, n_iter=2000, clip_eps=1e-9):
    m, n = X.shape
    X_b = np.hstack([np.ones((m,1)), X])   # (m, n+1)
    theta = np.zeros((n+1, 1))
    y_col = y.reshape(-1,1)

    costs = []
    for it in range(n_iter):
        Z = X_b @ theta
        A = sigmoid(Z)
        A = np.clip(A, clip_eps, 1-clip_eps)

        cost = - (1/m) * np.sum(y_col*np.log(A) + (1-y_col)*np.log(1-A))
        costs.append(cost)

        grad = (1/m) * (X_b.T @ (A - y_col))
        theta -= lr * grad

        # imprimir cada 200 iteraciones
        if it % (n_iter//10) == 0:
            print(f"Iter {it:4d}, cost={cost:.6f}")
    return theta, costs

theta, costs = train_logistic(Xs, y, lr=0.05, n_iter=2000)

# Predicciones y métricas
def predict_proba(X, theta):
    X_b = np.hstack([np.ones((X.shape[0],1)), X])
    return sigmoid(X_b @ theta).ravel()

probs = predict_proba(Xs, theta)
preds = (probs >= 0.5).astype(int)

cm = confusion_matrix(y, preds)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y, preds)
prec = precision_score(y, preds)
rec = recall_score(y, preds)
f1 = f1_score(y, preds)

print("Pesos:", theta.ravel())
print("Probabilidades:", probs)
print("Accuracy:", acc, "Precision:", prec, "Recall:", rec, "F1:", f1)
print("TP,TN,FP,FN:", tp, tn, fp, fn)

# Graficar coste
plt.plot(costs)
plt.xlabel("Iteraciones")
plt.ylabel("Cost")
plt.grid(True)
plt.show()
