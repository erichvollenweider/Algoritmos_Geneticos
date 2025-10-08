import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix


# ====== (a) ======
df = pd.read_csv("./Practico4/ScreenTime_vs_MentalWellness.csv")

# Separar características (X) y etiqueta (y)
X = df.drop("sleep_quality_1_5", axis=1)
y = df["sleep_quality_1_5"]

# Convertir variables categóricas en variables numéricas
X_encoded = pd.get_dummies(X, drop_first=True)

print(X_encoded.head())

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)


# ====== (b) ======
sns.boxplot(x="sleep_quality_1_5", y="screen_time_hours", data=df)


# ====== (c) ======
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ====== (d) ======
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    filled=True,
    fontsize=8,
    feature_names=X_encoded.columns,
    class_names=[str(c) for c in sorted(y.unique())]
)
plt.show()


importances = pd.Series(clf.feature_importances_, index=X_encoded.columns)
print(importances.sort_values(ascending=False).head(10))