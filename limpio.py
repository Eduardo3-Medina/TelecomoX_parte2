import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el DataFrame codificado (ya tiene One-Hot Encoding)
df = pd.read_parquet("df_final_limpio_codificado.parquet")

# --------------------------------------------
# RECONSTRUIR columna 'Contract' original
# --------------------------------------------
df['TipoContrato'] = 'Month-to-month'  # Base (se elimina con drop_first=True)

# Ajustar según las columnas dummy presentes
if 'Contract_One year' in df.columns:
    df.loc[df['Contract_One year'] == 1, 'TipoContrato'] = 'One year'
if 'Contract_Two year' in df.columns:
    df.loc[df['Contract_Two year'] == 1, 'TipoContrato'] = 'Two year'

# --------------------------------------------
# 1. Gráfico: Tipo de contrato vs Cancelación
# --------------------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='TipoContrato', hue='Churn', palette='Set2')
plt.title('Cancelación por tipo de contrato')
plt.xlabel('Tipo de contrato')
plt.ylabel('Cantidad de clientes')
plt.legend(title='Canceló')
plt.tight_layout()
plt.show()

# --------------------------------------------
# 2. Gráfico: Gasto total vs Cancelación (Boxplot)
# --------------------------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Churn', y='Charges.Total', palette='Set1')
plt.title('Distribución del gasto total según cancelación')
plt.xlabel('Canceló')
plt.ylabel('Gasto total')
plt.tight_layout()
plt.show()

# --------------------------------------------
# 3. Gráfico: Permanencia vs Gasto total (Scatterplot)
# --------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='tenure', y='Charges.Total', hue='Churn', palette='coolwarm', alpha=0.6)
plt.title('Relación entre permanencia, gasto total y cancelación')
plt.xlabel('Meses de permanencia')
plt.ylabel('Gasto total')
plt.legend(title='Canceló')
plt.tight_layout()
plt.show()
