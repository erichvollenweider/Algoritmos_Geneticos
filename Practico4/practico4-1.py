import numpy as np

from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler


# Dos puntos (personas) con características: [altura en cm, peso en kg, edad en años]
p1 = np.array([1, 2]) # persona A
p2 = np.array([2, 3]) # persona A
p3 = np.array([3, 3]) # persona B
p4 = np.array([5, 1]) # persona B
p5 = np.array([6, 2]) # persona B

# Punto a clasificar
q = np.array([3, 2])

# Distancia Euclidiana sin normalización
dist_raw1 = euclidean(q, p1)
dist_raw2 = euclidean(q, p2)
dist_raw3 = euclidean(q, p3)
dist_raw4 = euclidean(q, p4)
dist_raw5 = euclidean(q, p5)

print("Distancias Euclidianas sin normalizar:")
print("Q - P1:", dist_raw1)
print("Q - P2:", dist_raw2)
print("Q - P3:", dist_raw3)
print("Q - P4:", dist_raw4)
print("Q - P5:", dist_raw5)


# -----------------------------
# Normalización Z-score
# -----------------------------

# Armamos matriz con TODOS los puntos + q
data = np.array([p1, p2, p3, p4, p5, q])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Extraemos puntos normalizados
p1_s, p2_s, p3_s, p4_s, p5_s, q_s = data_scaled

# Distancias normalizadas
dist_scaled1 = euclidean(q_s, p1_s)
dist_scaled2 = euclidean(q_s, p2_s)
dist_scaled3 = euclidean(q_s, p3_s)
dist_scaled4 = euclidean(q_s, p4_s)
dist_scaled5 = euclidean(q_s, p5_s)

print("\nDistancias Euclidianas con normalización Z-score:")
print("Q - P1:", dist_scaled1)
print("Q - P2:", dist_scaled2)
print("Q - P3:", dist_scaled3)
print("Q - P4:", dist_scaled4)
print("Q - P5:", dist_scaled5)
