## Pipeline de Modelagem - comparação, tuning e avaliação final

## ATENÇÃO - DATA LEAKAGE CONFIRMADO E TRATADO:
# As colunas abaixo foram EXCLUÍDAS por vazarem o resultado da mesma avaliação que gera o target (ver análise: target bate 100% com taxa_alfabetizacao >= meta_alfabetizacao):
#    - taxa_alfabetizacao
#    - media_portugues
#    - proporcao_aluno_nivel_0 .. proporcao_aluno_nivel_8
#    - diferenca

## Colunas sem informação (variância zero ou 100% nulas), também excluídas:
#    - serie, rede, ano, ano_meta (valor único em toda a base)
#    - va_agropecuaria/industria/servicos_ultimo_disponivel (100% nulas)
#    - id_municipio, _ingested_at (identificador / metadado)

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

pd.set_option("display.max_columns", None)

# -----------------------------------------------------------------------
# 1. CARREGAR DADOS
# -----------------------------------------------------------------------
df = pd.read_csv("data/dataset_fase3_final.csv")

TARGET_COL = "target"

NUMERIC_COLS = [
    "meta_alfabetizacao",
    "num_escolas",
    "total_matriculas_anos_iniciais",
    "total_matriculas_2_ano",
    "total_docentes_anos_iniciais",
    "pct_agua_potavel",
    "pct_energia_rede_publica",
    "pct_esgoto_rede_publica",
    "pct_internet",
    "pct_biblioteca",
    "pct_laboratorio_informatica",
    "pct_quadra_esportes",
    "populacao",
    "pib_ultimo_disponivel",
]
CATEGORICAL_COLS = []  # nenhuma categórica confiável restante nesta versão

X = df[NUMERIC_COLS]
y = df[TARGET_COL]

# -----------------------------------------------------------------------
# 2. SPLIT TREINO / TESTE
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO
# -----------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_COLS),
])

# -----------------------------------------------------------------------
# 4. COMPARAR MODELOS CANDIDATOS COM VALIDAÇÃO CRUZADA
# -----------------------------------------------------------------------
modelos = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "random_forest": RandomForestClassifier(
        random_state=42, class_weight="balanced"
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    # Gradient Boosting não tem class_weight nativo; se quiser balancear,
    # use sample_weight no fit ou troque por HistGradientBoostingClassifier,
    # que aceita class_weight="balanced".
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

print("=" * 70)
print("COMPARAÇÃO DE MODELOS (validação cruzada, 5 folds)")
print("=" * 70)

for nome, modelo in modelos.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", modelo),
    ])
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True
    )
    print(f"\n--- {nome} ---")
    for metric in scoring:
        test_mean = scores[f"test_{metric}"].mean()
        train_mean = scores[f"train_{metric}"].mean()
        gap = train_mean - test_mean
        alerta = "  <-- possível overfitting" if gap > 0.1 else ""
        print(f"{metric}: test={test_mean:.3f} | train={train_mean:.3f}{alerta}")

# -----------------------------------------------------------------------
# 5. TUNING DO MELHOR MODELO (ajuste conforme o vencedor da comparação)
# -----------------------------------------------------------------------
pipeline_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
])

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10, 15],
    "classifier__min_samples_leaf": [1, 2, 5],
}

grid_search = GridSearchCV(pipeline_rf, param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)
print("\nMelhores parâmetros:", grid_search.best_params_)

modelo_final = grid_search.best_estimator_

# -----------------------------------------------------------------------
# 6. AVALIAÇÃO FINAL NO TESTE
# -----------------------------------------------------------------------
y_pred = modelo_final.predict(X_test)
y_proba = modelo_final.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
print("=" * 70)
print(classification_report(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 3))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------------------------------------------
# 7. SALVAR MODELO + CONJUNTO DE TESTE (para o script de interpretabilidade)
# -----------------------------------------------------------------------
joblib.dump(modelo_final, "reports/modelo_final.joblib")
X_test.to_csv("reports/X_test.csv", index=False)
y_test.to_csv("reports/y_test.csv", index=False)
print("\nModelo e conjunto de teste salvos em reports/ "
      "(usados por src/evaluation/interpretabilidade_shap.py)")