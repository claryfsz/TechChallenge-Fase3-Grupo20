# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = r"C:\Users\luiz.alves\TechChallenge-Fase3-Grupo20"
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "dataset_fase3_final.csv"))
print(f"Dataset carregado: {df.shape[0]} municípios, {df.shape[1]} colunas")
print()

LEAKAGE_COLS = [
    "taxa_alfabetizacao", "meta_alfabetizacao", "diferenca", "media_portugues",
    "proporcao_aluno_nivel_0", "proporcao_aluno_nivel_1", "proporcao_aluno_nivel_2",
    "proporcao_aluno_nivel_3", "proporcao_aluno_nivel_4", "proporcao_aluno_nivel_5",
    "proporcao_aluno_nivel_6", "proporcao_aluno_nivel_7", "proporcao_aluno_nivel_8",
]
LEAKAGE_COLS = [c for c in LEAKAGE_COLS if c in df.columns]

print("=" * 70)
print("COLUNAS DE DATA LEAKAGE (não usar como feature preditiva):")
print(LEAKAGE_COLS)
print("=" * 70)
print()

FEATURE_COLS = [
    "num_escolas", "total_matriculas_anos_iniciais", "total_matriculas_2_ano",
    "total_docentes_anos_iniciais", "pct_agua_potavel", "pct_energia_rede_publica",
    "pct_esgoto_rede_publica", "pct_internet", "pct_biblioteca",
    "pct_laboratorio_informatica", "pct_quadra_esportes", "populacao",
    "pib_ultimo_disponivel", "va_agropecuaria_ultimo_disponivel",
    "va_industria_ultimo_disponivel", "va_servicos_ultimo_disponivel",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

balance = df["target"].value_counts(normalize=True).sort_index()
balance_abs = df["target"].value_counts().sort_index()
print("Balanceamento do target:")
print(balance_abs)
print(balance.round(3))
print()

fig, ax = plt.subplots(figsize=(5, 4))
balance_abs.plot(kind="bar", ax=ax, color=["#B5651D", "#0E7C86"])
ax.set_xticklabels(["Não atingiu (0)", "Atingiu (1)"], rotation=0)
ax.set_ylabel("Número de municípios")
ax.set_title("Balanceamento do target")
for i, v in enumerate(balance_abs):
    ax.text(i, v + 5, str(v), ha="center")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "target_balance.png"), dpi=150)
plt.close()
print("Salvo: images/target_balance.png")
print()

n_cols = 4
n_rows = -(-len(FEATURE_COLS) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(FEATURE_COLS):
    sns.boxplot(data=df, x="target", y=col, ax=axes[i])
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["Não atingiu", "Atingiu"])
for j in range(len(FEATURE_COLS), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "boxplots_features_vs_target.png"), dpi=150)
plt.close()
print("Salvo: images/boxplots_features_vs_target.png")
print()

corr = df[FEATURE_COLS + ["target"]].corr(numeric_only=True)["target"].drop("target")
corr = corr.sort_values()
print("Correlação de cada feature com o target (sem colunas de leakage):")
print(corr.round(3))
print()

fig, ax = plt.subplots(figsize=(7, max(4, len(corr) * 0.4)))
colors = ["#B5651D" if v < 0 else "#0E7C86" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlação das features com o target")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "correlacao_target.png"), dpi=150)
plt.close()
print("Salvo: images/correlacao_target.png")
print()

print("Estatísticas de população:")
print(df["populacao"].describe())
print()

pequenos = df[df["populacao"] < 5000]
print(f"Municípios com população < 5.000: {len(pequenos)} "
      f"({len(pequenos) / len(df) * 100:.1f}% da base)")
print("Taxa de atingimento da meta nesse grupo:",
      round(pequenos["target"].mean(), 3),
      "vs. geral:", round(df["target"].mean(), 3))
print()

print("Salvo: reports/eda_insights.md")
print("Análise concluída.")