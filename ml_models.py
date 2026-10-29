
# benchmark_modelos.py

import warnings

import numpy as np
import pandas as pd

from scipy.special import expit
from scipy.linalg import pinv

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


# ============================================================
# 1. EXTREME LEARNING MACHINE (ELM)
#    Baseada em Al-Shamiri et al. (2019)
# ============================================================

class ELMRegressor(RegressorMixin, BaseEstimator):
    """
    Extreme Learning Machine para regressão.

    Implementação baseada na metodologia descrita no artigo:
    - Uma camada oculta.
    - Ativação sigmoide logística.
    - Pesos de entrada e vieses gerados por U(-1, 1).
    - Pesos de saída calculados pela pseudoinversa
      de Moore-Penrose.
    """

    def __init__(self, n_hidden=110, random_state=42):
        self.n_hidden = n_hidden
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(
            X,
            y,
            y_numeric=True,
            ensure_all_finite=True
        )

        if self.n_hidden < 1:
            raise ValueError("n_hidden deve ser maior que zero.")

        rng = np.random.default_rng(self.random_state)

        n_features = X.shape[1]

        # Pesos de entrada e vieses sorteados uniformemente
        # no intervalo [-1, 1].
        self.weights_hidden_ = rng.uniform(
            -1.0,
            1.0,
            size=(n_features, self.n_hidden)
        )

        self.bias_hidden_ = rng.uniform(
            -1.0,
            1.0,
            size=self.n_hidden
        )

        # Saída da camada oculta com ativação sigmoide.
        Z = X @ self.weights_hidden_ + self.bias_hidden_
        H = expit(Z)

        # Cálculo dos pesos de saída pela pseudoinversa.
        self.weights_output_ = pinv(H) @ y

        self.n_features_in_ = n_features

        return self

    def predict(self, X):
        check_is_fitted(
            self,
            attributes=[
                "weights_hidden_",
                "bias_hidden_",
                "weights_output_"
            ]
        )

        X = check_array(X, ensure_all_finite=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Esperadas {self.n_features_in_} variáveis de entrada, "
                f"mas recebidas {X.shape[1]}."
            )

        Z = X @ self.weights_hidden_ + self.bias_hidden_
        H = expit(Z)

        return H @ self.weights_output_


# ============================================================
# 2. CARREGAMENTO E VERIFICAÇÃO DOS DADOS
# ============================================================

arquivo = "concrete_features.csv"
alvo = "Compressive_strength"
SEED = 42

df = pd.read_csv(arquivo)

if alvo not in df.columns:
    raise ValueError(
        f"A coluna alvo '{alvo}' não foi encontrada."
    )

if df.isnull().any().any():
    raise ValueError(
        "Existem valores ausentes. Trate-os antes de comparar "
        "os modelos."
    )

X = df.drop(columns=[alvo])
y = df[alvo]

# Este benchmark pressupõe entradas numéricas.
if not all(
    pd.api.types.is_numeric_dtype(dtype)
    for dtype in X.dtypes
):
    raise ValueError(
        "Existem variáveis de entrada não numéricas. "
        "Codifique-as antes de executar este benchmark."
    )

if not pd.api.types.is_numeric_dtype(y.dtype):
    raise ValueError("A variável alvo deve ser numérica.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError(
        "Existem valores infinitos ou inválidos nas entradas."
    )

if not np.isfinite(y.to_numpy(dtype=float)).all():
    raise ValueError(
        "Existem valores infinitos ou inválidos no alvo."
    )

print(f"Observações: {len(df)}")
print(f"Variáveis de entrada: {X.shape[1]}")
print(f"Alvo: {alvo}")


# ============================================================
# 3. DEFINIÇÃO DOS MODELOS
# ============================================================

modelos = {
    # Referências lineares
    "Regressão Linear": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    # Árvores individuais e ensembles
    "Decision Tree": DecisionTreeRegressor(
        random_state=SEED
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=SEED,
        n_jobs=1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=SEED,
        n_jobs=1
    ),

    # Boosting
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=SEED
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        random_state=SEED
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=1,
        verbosity=0
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=SEED,
        thread_count=1,
        verbose=False,
        allow_writing_files=False
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        n_jobs=1,
        verbosity=-1
    ),

    # Métodos baseados em distância e margem
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(
            n_neighbors=5,
            weights="distance"
        )
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

    # Rede neural treinada por backpropagation
    "MLP Regressor": make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            random_state=SEED
        )
    ),

    # ELM baseada no artigo
    # O MinMaxScaler é ajustado dentro de cada fold.
    "ELM": make_pipeline(
        MinMaxScaler(feature_range=(0, 1)),
        ELMRegressor(
            n_hidden=110,
            random_state=SEED
        )
    )
}


# ============================================================
# 4. VALIDAÇÃO CRUZADA
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

metricas = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

resultados = []

for nome, modelo in modelos.items():
    print(f"Avaliando: {nome}...")

    try:
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")

            scores = cross_validate(
                modelo,
                X,
                y,
                cv=cv,
                scoring=metricas,
                n_jobs=-1,
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
            "R² desvio": scores["test_R2"].std(ddof=1),
            "Tempo médio (s)": scores["fit_time"].mean(),
            "Status": "OK"
        })

        if avisos:
            print(
                f"  Aviso: {len(avisos)} aviso(s) durante "
                "a avaliação. Verifique se necessário."
            )

    except Exception as erro:
        print(f"  ERRO em {nome}: {erro}")

        resultados.append({
            "Modelo": nome,
            "MAE médio": np.nan,
            "MAE desvio": np.nan,
            "RMSE médio": np.nan,
            "RMSE desvio": np.nan,
            "R² médio": np.nan,
            "R² desvio": np.nan,
            "Tempo médio (s)": np.nan,
            "Status": f"ERRO: {erro}"
        })


# ============================================================
# 5. ORGANIZAÇÃO E EXPORTAÇÃO
# ============================================================

tabela = pd.DataFrame(resultados)

tabela = tabela.sort_values(
    by="RMSE médio",
    ascending=True,
    na_position="last"
).reset_index(drop=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.precision", 5)

print("\nRESULTADOS DA VALIDAÇÃO CRUZADA")
print(tabela.to_string(index=False))

saida = "resultados_comparacao_modelos_features.csv"

tabela.to_csv(
    saida,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nResultados salvos em {saida}")
