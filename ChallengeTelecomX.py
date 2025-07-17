import pandas as pd
from pandas import json_normalize
ruta_json = r'C:\Users\UNSAAC\Desktop\facil sirve despues\ORACLE\ChallengeTelecomX\TelecomX_Data.json'
df = pd.read_json(ruta_json)  
customer_df = json_normalize(df['customer'])
phone_df = json_normalize(df['phone'])
internet_df = json_normalize(df['internet'])
account_df = json_normalize(df['account'])
charges_df = json_normalize(df['account'].apply(lambda x: x['Charges']))

df_final = pd.concat([
    df[['customerID', 'Churn']].reset_index(drop=True),
    customer_df,
    phone_df,
    internet_df,
    account_df,
    charges_df 
], axis=1)

columnas_deseadas = [
    'customerID','Churn','gender','SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'Charges.Monthly', 'Charges.Total'  
]
'''print(df_final[columnas_deseadas].iloc[0])'''
'''
print(df_final.columns)
print(df_final.dtypes)
print(df_final['customer'].iloc[0])
print(df_final['phone'].iloc[0])
print(df_final['internet'].iloc[0])
print(df_final['account'].iloc[0])
'''
#CONVIRTIENDO LOS DATOS

##CONVERTIR A ENTEROS
df_final['SeniorCitizen'] = df_final['SeniorCitizen'].astype(int)
df_final['tenure'] = df_final['tenure'].astype(int)

##CONVERTIR A FLOAT
df_final['Charges.Monthly'] = pd.to_numeric(df_final['Charges.Monthly'], errors='coerce').fillna(0)
df_final['Charges.Total'] = pd.to_numeric(df_final['Charges.Total'], errors='coerce').fillna(0)

##CONVERTIR A BOOLEANOS
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
##CONVERTIR A STRING
df_final['InternetService'] = df_final['InternetService'].astype("string")
df_final['Contract'] = df_final['Contract'].astype("string")
df_final['PaymentMethod'] = df_final['PaymentMethod'].astype("string")
df_final['gender'] = df_final['gender'].astype("string")
df_final['customerID'] = df_final['customerID'].astype("string")

print(df_final.dtypes)
print(df_final[columnas_deseadas].iloc[0])

df_final.to_parquet("df_final_limpio.parquet")  # Recomendado: mantiene tipos
