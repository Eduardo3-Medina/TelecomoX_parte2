import pandas as pd
dfParquet = pd.read_parquet(r'C:\Users\UNSAAC\Desktop\facil sirve despues\ORACLE\ChallengeTelecomX\df_final_limpio.parquet');
columnas_deseadas = [
    'customerID','Churn','gender','SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'Charges.Monthly', 'Charges.Total'  
]
print(dfParquet[columnas_deseadas].iloc[0])
print("\n Valores vacios?")
print(dfParquet.isnull().sum())
print("\n Valores Duplicados?")
duplicados = dfParquet['customerID'].duplicated()
print(duplicados.sum())