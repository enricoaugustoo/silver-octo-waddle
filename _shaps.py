import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

# Carregar os dados
df = pd.read_csv("concrete_features.csv")

X = df.drop(columns=["Compressive_strength"])
y = df["Compressive_strength"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar o modelo
modelo_extra = ExtraTreesRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
modelo_extra.fit(X_train, y_train)

# Calcular os valores SHAP
explainer = shap.TreeExplainer(modelo_extra)
shap_values = explainer(X_test)

# Gráficos
# Gráfico de importância global
shap.plots.bar(shap_values, show=False, max_display=20)
plt.gcf().set_size_inches(11, 7)
plt.gcf().subplots_adjust(left=0.40)
plt.show()

# Gráfico beeswarm
shap.plots.beeswarm(shap_values, show=False, max_display=20)
plt.gcf().set_size_inches(12, 8)
plt.gcf().subplots_adjust(left=0.40)
plt.show()