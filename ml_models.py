
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor


# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

arquivo = "concrete_features.csv"
alvo = "Compressive_strength"

df = pd.read_csv(arquivo)

# Verificações básicas
if alvo not in df.columns:
    raise ValueError(f"A coluna alvo '{alvo}' não foi encontrada.")

if df.isnull().any().any():
    raise ValueError(
        "Existem valores ausentes. Trate-os antes de comparar os modelos."
    )

if not np.isfinite(df.select_dtypes(include=np.number)).all().all():
    raise ValueError("Existem valores numéricos infinitos ou inválidos.")

X = df.drop(columns=[alvo])
y = df[alvo]

print(f"Observações: {len(df)}")
print(f"Variáveis de entrada: {X.shape[1]}")
print(f"Alvo: {alvo}")


# ============================================================
# 2. DEFINIÇÃO DOS MODELOS
# ============================================================

modelos = {
    "Regressão Linear": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),

    "SVR - RBF": make_pipeline(
        StandardScaler(),
        SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
            gamma="scale"
        )
    ),

    "MLP Regressor": make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
    )
}


# ============================================================
# 3. VALIDAÇÃO CRUZADA
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

metricas = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

resultados = []

for nome, modelo in modelos.items():
    print(f"Avaliando: {nome}...")

    scores = cross_validate(
        modelo,
        X,
        y,
        cv=cv,
        scoring=metricas,
        n_jobs=1,
        return_train_score=False,
        error_score="raise"
    )

    resultados.append({
        "Modelo": nome,
        "MAE médio": -scores["test_MAE"].mean(),
        "MAE desvio": scores["test_MAE"].std(ddof=1),
        "RMSE médio": -scores["test_RMSE"].mean(),
        "RMSE desvio": scores["test_RMSE"].std(ddof=1),
        "R² médio": scores["test_R2"].mean(),
        "R² desvio": scores["test_R2"].std(ddof=1)
    })


# ============================================================
# 4. ORGANIZAÇÃO E EXPORTAÇÃO DOS RESULTADOS
# ============================================================

tabela = pd.DataFrame(resultados)

# Ordenar pelo menor MAE médio
tabela = tabela.sort_values(
    by="MAE médio",
    ascending=True
).reset_index(drop=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.precision", 4)

print("\nRESULTADOS DA VALIDAÇÃO CRUZADA")
print(tabela.to_string(index=False))

tabela.to_csv(
    "resultados_comparacao_modelos.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nResultados salvos em resultados_comparacao_modelos.csv")
