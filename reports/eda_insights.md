# EDA - Resumo para o README (seção "Insights encontrados")

*Gerado automaticamente a partir de `dataset_fase3_final.csv`.*

## Balanceamento do target

- Atingiram a meta: 2788 municípios (53.3%)
- Não atingiram: 2444 municípios (46.7%)

## Correlação com o target (features preditivas, sem leakage)

**Top 3 associações positivas:**
- `pct_esgoto_rede_publica`: 0.068
- `pct_internet`: 0.057
- `pct_energia_rede_publica`: 0.043

**Top 3 associações negativas:**
- `num_escolas`: -0.060
- `total_matriculas_2_ano`: -0.038
- `total_matriculas_anos_iniciais`: -0.037

## Municípios pequenos

- 1169 municípios (22.3%) têm população < 5.000 habitantes.
- Taxa de atingimento da meta nesse grupo: 0.571 vs. 0.533 na base geral.
