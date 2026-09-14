## Interpretabilidade do modelo final via SHAP

## Requer que src/modeling/pipeline_modelagem_treino.py já tenha rodado
# (usa reports/modelo_final.joblib, reports/X_test.csv).

## Gera:
# - images/shap_summary.png
# - ranking de importância média impresso no terminal

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

modelo_final = joblib.load("reports/modelo_final.joblib")
X_test = pd.read_csv("reports/X_test.csv")

NUMERIC_COLS = list(X_test.columns)

classifier = modelo_final.named_steps["classifier"]
X_test_transformed = modelo_final.named_steps["preprocessor"].transform(X_test)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_transformed)

# Versões diferentes do shap retornam formatos diferentes:
# - lista [array_classe0, array_classe1]  (versões mais antigas)
# - array 3D (amostras, features, classes)  (versões mais novas)
# - array 2D (amostras, features)  (quando só há uma saída)
if isinstance(shap_values, list):
    valores_classe_1 = shap_values[1]
elif shap_values.ndim == 3:
    valores_classe_1 = shap_values[:, :, 1]
else:
    valores_classe_1 = shap_values

plt.figure()
shap.summary_plot(
    valores_classe_1,
    X_test_transformed,
    feature_names=NUMERIC_COLS,
    show=False,
)
plt.tight_layout()
plt.savefig("images/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico SHAP salvo em: images/shap_summary.png")

# Ranking simples de importância média absoluta (bom pro README/relatório)
importancia_media = np.abs(valores_classe_1).mean(axis=0)
ranking = pd.Series(importancia_media, index=NUMERIC_COLS).sort_values(ascending=False)
print("\nRanking de importância média (SHAP):")
print(ranking)