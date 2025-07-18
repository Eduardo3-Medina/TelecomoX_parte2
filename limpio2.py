import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar datos desde archivo parquet
df = pd.read_parquet("df_final_limpio_codificado.parquet")  # Reemplaza con el nombre real

# 2. Mostrar tipos para verificar columnas problemáticas
print("Tipos de datos originales:\n", df.dtypes)

# 3. Eliminar columnas irrelevantes si existen
columnas_a_eliminar = ['customerID'] if 'customerID' in df.columns else []
df = df.drop(columns=columnas_a_eliminar)

# 4. Convertir variables categóricas en variables dummy
df = pd.get_dummies(df, drop_first=True)

# 5. Separar features y target
y_col = [col for col in df.columns if "Churn" in col][0]  # Detecta automáticamente si se llama 'Churn' o 'Churn_Yes'
X = df.drop(y_col, axis=1)
y = df[y_col]

# 6. Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Escalar los datos numéricos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Entrenar modelos
modelos = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)

    resultados[nombre] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Reporte": classification_report(y_test, y_pred, zero_division=0)
    }

# 9. Mostrar métricas
for nombre, metricas in resultados.items():
    print(f"\nModelo: {nombre}")
    print("Accuracy:", round(metricas["Accuracy"], 3))
    print("Precision:", round(metricas["Precision"], 3))
    print("Recall:", round(metricas["Recall"], 3))
    print("F1 Score:", round(metricas["F1 Score"], 3))
    print("Confusion Matrix:\n", metricas["Confusion Matrix"])
    print("Reporte de Clasificación:\n", metricas["Reporte"])

# 10. Visualizar matriz de confusión del mejor modelo
modelo_mejor = max(resultados, key=lambda m: resultados[m]["F1 Score"])
matriz_mejor = resultados[modelo_mejor]["Confusion Matrix"]

plt.figure(figsize=(5, 4))
sns.heatmap(matriz_mejor, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusión - {modelo_mejor}")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
