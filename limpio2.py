# 1. Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 2. Cargar y preparar datos
df = pd.read_parquet("df_final_limpio_codificado.parquet")

columnas_a_eliminar = ['customerID'] if 'customerID' in df.columns else []
df = df.drop(columns=columnas_a_eliminar)

df = pd.get_dummies(df, drop_first=True)

y_col = [col for col in df.columns if "Churn" in col][0]
X = df.drop(y_col, axis=1)
y = df[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Definir modelos
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(kernel='linear', probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

resultados = {}

# 4. Entrenar y evaluar modelos
for nombre, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)

    resultados[nombre] = {
        "modelo": modelo,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Reporte": classification_report(y_test, y_pred, zero_division=0)
    }

# 5. Mostrar métricas
for nombre, metricas in resultados.items():
    print(f"\nModelo: {nombre}")
    print("Accuracy:", round(metricas["Accuracy"], 3))
    print("Precision:", round(metricas["Precision"], 3))
    print("Recall:", round(metricas["Recall"], 3))
    print("F1 Score:", round(metricas["F1 Score"], 3))
    print("Confusion Matrix:\n", metricas["Confusion Matrix"])
    print("Reporte de Clasificación:\n", metricas["Reporte"])

# 6. Visualizar matriz de confusión del mejor modelo
modelo_mejor = max(resultados, key=lambda m: resultados[m]["F1 Score"])
matriz_mejor = resultados[modelo_mejor]["Confusion Matrix"]

plt.figure(figsize=(5, 4))
sns.heatmap(matriz_mejor, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusión - {modelo_mejor}")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# 7. Análisis de variables más relevantes
# A. Regresión Logística
coef_lr = pd.Series(resultados["Logistic Regression"]["modelo"].coef_[0], index=X.columns)
top_coef_lr = coef_lr.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_coef_lr.plot(kind='barh', color='skyblue')
plt.title("Top 10 Variables - Regresión Logística")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# B. Random Forest
importancias_rf = pd.Series(resultados["Random Forest"]["modelo"].feature_importances_, index=X.columns)
top_rf = importancias_rf.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_rf.plot(kind='barh', color='orange')
plt.title("Top 10 Variables - Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# C. SVM (coeficientes)
coef_svm = pd.Series(resultados["SVM"]["modelo"].coef_[0], index=X.columns)
top_coef_svm = coef_svm.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_coef_svm.plot(kind='barh', color='green')
plt.title("Top 10 Variables - SVM")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# D. XGBoost
importancias_xgb = pd.Series(resultados["XGBoost"]["modelo"].feature_importances_, index=X.columns)
top_xgb = importancias_xgb.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_xgb.plot(kind='barh', color='red')
plt.title("Top 10 Variables - XGBoost")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# E. KNN (Permutation Importance)
perm_knn = permutation_importance(resultados["KNN"]["modelo"], X_test_scaled, y_test, n_repeats=10, random_state=42)
importancia_perm_knn = pd.Series(perm_knn.importances_mean, index=X.columns)
top_knn = importancia_perm_knn.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_knn.plot(kind='barh', color='purple')
plt.title("Top 10 Variables - KNN (Permutation Importance)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
