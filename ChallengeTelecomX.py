import pandas as pd
from pandas import json_normalize

# Cargar el archivo JSON
ruta_json = r'C:\Users\UNSAAC\Desktop\facil sirve despues\ORACLE\ChallengeTelecomX\TelecomX_Data.json'
df = pd.read_json(ruta_json)

# Normalizar las secciones del JSON
customer_df = json_normalize(df['customer'])
phone_df = json_normalize(df['phone'])
internet_df = json_normalize(df['internet'])
account_df = json_normalize(df['account'])
charges_df = json_normalize(df['account'].apply(lambda x: x['Charges']))

# Concatenar en un solo DataFrame
df_final = pd.concat([
    df[['customerID', 'Churn']].reset_index(drop=True),
    customer_df,
    phone_df,
    internet_df,
    account_df,
    charges_df 
], axis=1)

# Selección de columnas relevantes
columnas_deseadas = [
    'customerID','Churn','gender','SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'Charges.Monthly', 'Charges.Total'  
]

# Mostrar columnas eliminadas
columnas_actuales = df_final.columns.tolist()
columnas_eliminadas = [col for col in columnas_actuales if col not in columnas_deseadas]
print("Columnas eliminadas:", columnas_eliminadas)

# Filtrar solo las columnas deseadas
df_final = df_final[columnas_deseadas]

# Conversión de tipos
df_final['SeniorCitizen'] = df_final['SeniorCitizen'].astype(int)
df_final['tenure'] = df_final['tenure'].astype(int)

df_final['Charges.Monthly'] = pd.to_numeric(df_final['Charges.Monthly'], errors='coerce').fillna(0)
df_final['Charges.Total'] = pd.to_numeric(df_final['Charges.Total'], errors='coerce').fillna(0)

df_final['Partner'] = df_final['Partner'].astype(str).str.lower().map({'yes': True, 'no': False})
df_final['Dependents'] = df_final['Dependents'].astype(str).str.lower().map({'yes': True, 'no': False})
df_final['PhoneService'] = df_final['PhoneService'].astype(str).str.lower().map({'yes': True, 'no': False})
df_final['MultipleLines'] = df_final['MultipleLines'].apply(lambda x: True if x == 'Yes' else False)
df_final['OnlineSecurity'] = df_final['OnlineSecurity'].apply(lambda x: True if x == 'Yes' else False)
df_final['OnlineBackup'] = df_final['OnlineBackup'].apply(lambda x: True if x == 'Yes' else False)
df_final['DeviceProtection'] = df_final['DeviceProtection'].apply(lambda x: True if x == 'Yes' else False)
df_final['TechSupport'] = df_final['TechSupport'].apply(lambda x: True if x == 'Yes' else False)
df_final['StreamingTV'] = df_final['StreamingTV'].apply(lambda x: True if x == 'Yes' else False)
df_final['StreamingMovies'] = df_final['StreamingMovies'].apply(lambda x: True if x == 'Yes' else False)
df_final['PaperlessBilling'] = df_final['PaperlessBilling'].astype(str).str.lower().map({'yes': True, 'no': False})
df_final['Churn'] = df_final['Churn'].apply(lambda x: True if x == 'Yes' else False)

df_final['InternetService'] = df_final['InternetService'].astype("string")
df_final['Contract'] = df_final['Contract'].astype("string")
df_final['PaymentMethod'] = df_final['PaymentMethod'].astype("string")
df_final['gender'] = df_final['gender'].astype("string")
df_final['customerID'] = df_final['customerID'].astype("string")

# --- APLICAR ONE-HOT ENCODING A VARIABLES CATEGÓRICAS ---
columnas_categoricas = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
df_final = pd.get_dummies(df_final, columns=columnas_categoricas, drop_first=True)

# Ver tipos de datos finales
print(df_final.dtypes)

# Ver ejemplo de fila transformada
print(df_final.iloc[0])

# Guardar el DataFrame final
df_final.to_parquet("df_final_limpio_codificado.parquet", index = False)
