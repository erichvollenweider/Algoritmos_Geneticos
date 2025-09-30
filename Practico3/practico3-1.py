# 1)----------------------------
# 1.a).-------------------------
# pacientes = 200; enfermos = 40; no_emfermos = 160 
# clasificador detectó enfermos = 50 de los cuales 30 si eran
# TP = 30; TN = 140; FP = 20; FN = 10;


# TP (True Positive / Verdadero positivo): enfermos reales detectados como enfermos.
# FP (False Positive / Falso positivo): sanos detectados como enfermos.
# FN (False Negative / Falso negativo): enfermos detectados como sanos.
# TN (True Negative / Verdadero negativo): sanos detectados como sanos.


# 1.b).-------------------------
# Accuracy (Exactitud) = TP + TN / TP + TN + FP + FN = 30 + 140 / 30 + 140 + 20 + 10 = 0.85
# Precision (Presición) = TP / TP + FP = 30 / 30 + 20 = 0.6
# Recall (Exhaustividad) = TP / TP + FN = 30 / 30 + 10 = 0.75
# F1-Score = 2 ⋅ (Precision ⋅ Recall / Precision + Recall) = 2 ⋅ (0.6 ⋅ 075 / 0.6 + 0.75) = 0,66

# 1.c).-------------------------
