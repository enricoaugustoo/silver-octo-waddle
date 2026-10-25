import pandas as pd

arquivo = 'concrete_features.csv'

df = pd.read_csv(arquivo)

print(df.columns)

