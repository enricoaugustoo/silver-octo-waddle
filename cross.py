
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)

# 1. Carregar os dados
df = pd.read_csv('concrete_features.csv')

X = df.drop(columns=['Compressive_strength'])
y = df['Compressive_strength']

# 2. Definir os modelos
modelos = {
    'Regressão Linear': make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    'Extra Trees': ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    'HistGradientBoosting': HistGradientBoostingRegressor(
        max_iter=200,
        random_state=42
    )
}

# 3. Definir a validação cruzada
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Avaliar os modelos
resultados = []

for nome, modelo in modelos.items():

    scores = cross_validate(
        modelo,
        X,
        y,
        cv=cv,
        scoring={
            'MAE': 'neg_mean_absolute_error',
            'RMSE': 'neg_root_mean_squared_error',
            'R2': 'r2'
        },
        n_jobs=1
    )

    # Converter MAE e RMSE para valores positivos
    mae = -scores['test_MAE']
    rmse = -scores['test_RMSE']
    r2 = scores['test_R2']

    resultados.append({
        'Modelo': nome,
        'MAE médio': mae.mean(),
        'MAE desvio': mae.std(ddof=1),
        'RMSE médio': rmse.mean(),
        'RMSE desvio': rmse.std(ddof=1),
        'R² médio': r2.mean(),
        'R² desvio': r2.std(ddof=1)
    })

# 5. Exibir os resultados
resultados = pd.DataFrame(resultados)
resultados = resultados.sort_values('RMSE médio')

print(resultados.round(4).to_string(index=False))
