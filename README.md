# Tech Challenge – Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil

## 1. Contexto do Problema

A alfabetização infantil é um dos principais indicadores do desenvolvimento
educacional e social do Brasil. O programa federal Compromisso Nacional
Criança Alfabetizada estabelece metas anuais de percentual de estudantes
alfabetizados ao final do 2º ano do Ensino Fundamental para municípios e
estados, com o objetivo de que todas as crianças brasileiras estejam
alfabetizadas até 2030.

Apenas o resultado atual não é suficiente para orientar decisões
estratégicas: gestores públicos precisam antecipar riscos, identificar
municípios vulneráveis e entender quais fatores têm maior peso no desempenho
educacional. Este projeto irá transformar dados públicos em inteligência aplicada a essa decisão.

## 2. Objetivo Analítico

Desenvolver um modelo supervisionado de classificação binária capaz de
prever se um município atingiu ou não a meta de alfabetização estabelecida pelo programa, 
a partir de diferentes variáveis.

- **Target**: `1` se o município atingiu ou superou a meta
  (`taxa_alfabetizacao >= meta_alfabetizacao`), `0` caso contrário.
- **Unidade de análise**: município, ano-base 2024 (rede Municipal de ensino).

## 3. Descrição da Base Utilizada

A base parte da camada Gold construída na Fase 2 (Indicador Criança
Alfabetizada, metas nacionais/estaduais/municipais, dados territoriais),
enriquecida nesta fase com fontes externas:

| Fonte | Dataset | O que traz |
|---|---|---|
| INEP – Avaliação da Alfabetização | `br_inep_avaliacao_alfabetizacao` | Indicador real e metas de alfabetização por município/ano |
| INEP – Censo Escolar | `br_inep_censo_escolar` (tabela `escola`) | Matrículas, docentes e infraestrutura escolar por município/ano |
| IBGE – População | `br_ibge_populacao` | População total do município/ano |
| IBGE – PIB dos Municípios | `br_ibge_pib` | PIB total (último ano disponível) |

**Correção de qualidade de dados (Fase 2 → Fase 3):** a tabela original de
indicador apresentava linhas duplicadas por município/ano. A causa raiz foi
dupla: (1) a coluna `rede` (Estadual/Municipal/Pública combinada) nãoo havia
sido decodificada nem usada como chave de agregação; (2) o merge com a
tabela de metas gerava um efeito cartesiano por falta de deduplicação prévia.
A correção restringiu o indicador à rede Municipal e deduplicou a tabela de metas antes
do merge.

**Por que a base cobre apenas 2024:** as metas do programa são definidas
para 2024–2030, mas resultado real só existe para anos já ocorridos. 2024 é
o único ano em que meta e resultado real existem.

**Granularidade final:** 5.232 municípios × 1 ano (2024), sem duplicatas.

## 4. Etapas de Modelagem
<!-- Pessoa 2 -->

**Definição do target:** a base Gold da Fase 2 é agregada por
município/ano (não por aluno individual). O target foi definido como
binário: `1` se o município atingiu a meta oficial de alfabetização no
ano (`taxa_alfabetizacao >= meta_alfabetizacao`), `0` caso contrário.
Essa definição foi escolhida por já vir pronta na base (`diferenca`) e
por conectar diretamente com uma das perguntas de negócio do desafio
(prever risco de município não atingir metas futuras).

**Tratamento de data leakage:** foram identificadas e removidas do
conjunto de features todas as variáveis derivadas da mesma avaliação
que compõe o target:
- `taxa_alfabetizacao` — reproduz o target quase perfeitamente
  (`taxa_alfabetizacao >= meta_alfabetizacao` bate 100% com o target)
- `media_portugues` — nota média da mesma avaliação (Saeb)
- `proporcao_aluno_nivel_0` a `proporcao_aluno_nivel_8` — distribuição
  de proficiência da mesma avaliação (somam ~100% por linha)
- `diferenca` — usada diretamente para construir o target

Também foram descartadas colunas sem informação: `serie`, `rede`,
`ano` e `ano_meta` (valor único em toda a base, sem variância) e
`id_municipio`/`_ingested_at` (identificador e metadado, não são
features).

**Valores faltantes:** as colunas
`va_agropecuaria_ultimo_disponivel`, `va_industria_ultimo_disponivel`
e `va_servicos_ultimo_disponivel` estavam 100% nulas — o IBGE ainda
não publicou a abertura setorial do PIB para o último ano disponível.
Foram descartadas; o PIB total (`pib_ultimo_disponivel`) já cobre a
informação essencial. Para as demais variáveis numéricas, a
`SimpleImputer(strategy="median")` foi incluída na pipeline como
salvaguarda, embora a base final não tenha apresentado nulos nas
colunas usadas.

**Features finais utilizadas:** `meta_alfabetizacao`, `num_escolas`,
`total_matriculas_anos_iniciais`, `total_matriculas_2_ano`,
`total_docentes_anos_iniciais`, `pct_agua_potavel`,
`pct_energia_rede_publica`, `pct_esgoto_rede_publica`, `pct_internet`,
`pct_biblioteca`, `pct_laboratorio_informatica`,
`pct_quadra_esportes`, `populacao`, `pib_ultimo_disponivel`.

**Pipeline:** pré-processamento (`SimpleImputer` + `StandardScaler`)
integrado ao classificador via `sklearn.pipeline.Pipeline` +
`ColumnTransformer`, garantindo que o fit dos transformadores ocorra
apenas nos dados de treino.

**Split:** treino/teste com `train_test_split` (80/20), estratificado
pelo target para preservar a proporção das classes.

## 5. Escolha do Algoritmo
<!-- Pessoa 2 -->

**Modelos testados:** Regressão Logística (baseline linear), Random
Forest e Gradient Boosting, comparados via validação cruzada
estratificada (`StratifiedKFold`, 5 folds) nas métricas accuracy,
precision, recall, F1 e ROC AUC.

**Desbalanceamento de classes:** a primeira rodada mostrou recall
baixo para a classe 0 (município não atingiu a meta). Foi aplicado
`class_weight="balanced"` na Regressão Logística e no Random Forest,
o que elevou o recall da classe 0 de 32% para 53% com perda mínima de
accuracy geral (58% → 60%).

**Ajuste de hiperparâmetros:** `GridSearchCV` no Random Forest
(`n_estimators`, `max_depth`, `min_samples_leaf`), otimizando F1.
Melhor combinação: `max_depth=5, min_samples_leaf=2,
n_estimators=300`.

**Modelo final escolhido: Random Forest** — teve o melhor ROC AUC na
validação cruzada (0.644) entre os três candidatos, e o tuning
reduziu o overfitting observado na versão sem restrição de
profundidade (train AUC caiu de 1.00 para um comportamento mais
alinhado ao teste). Também foi priorizado por sua compatibilidade
direta com SHAP (`TreeExplainer`), atendendo ao requisito de
interpretabilidade do desafio.

## 6. Métricas de Avaliação
<!-- Pessoa 2 -->

**Validação cruzada (5 folds), Random Forest com `class_weight="balanced"`:**

| Métrica   | Teste (média CV) | Treino (média CV) |
|-----------|-------------------|---------------------|
| Accuracy  | 0.607             | 1.000 *(overfitting antes do tuning)* |
| Precision | 0.631             | 1.000 |
| Recall    | 0.631             | 1.000 |
| F1        | 0.631             | 1.000 |
| ROC AUC   | 0.644             | 1.000 |

O Random Forest sem restrição de profundidade memorizou o treino
(scores = 1.0). Após o `GridSearchCV` (`max_depth=5,
min_samples_leaf=2, n_estimators=300`), o modelo final ficou mais
conservador e generalizou melhor.

**Avaliação final no conjunto de teste (hold-out, modelo tunado):**

| Classe | Precision | Recall | F1   |
|--------|-----------|--------|------|
| 0 (não atingiu a meta) | 0.58 | 0.53 | 0.55 |
| 1 (atingiu a meta)     | 0.61 | 0.66 | 0.64 |

- Accuracy geral: 0.60
- ROC AUC: 0.625
- Matriz de confusão: `[[259, 230], [191, 367]]`

**Leitura honesta do resultado:** o desempenho é modesto (AUC ~0.62),
o que é esperado — depois de remover todas as variáveis que vazavam o
próprio resultado da avaliação (ver seção 4), restam apenas variáveis
de infraestrutura, porte do município e a meta em si, que têm poder
preditivo real, porém limitado, sobre o resultado educacional.

## 7. Interpretação dos Resultados
<!-- Pessoa 2 -->

Interpretabilidade obtida via SHAP (`TreeExplainer`) sobre o Random
Forest final. Gráfico completo em `images/shap_summary.png`.

**Ranking de importância média (SHAP):**

1. `meta_alfabetizacao`
2. `pct_esgoto_rede_publica`
3. `num_escolas`
4. `pib_ultimo_disponivel`
5. `total_matriculas_anos_iniciais`
6. `pct_agua_potavel`
7. `pct_internet`
8. `total_docentes_anos_iniciais`
9. `total_matriculas_2_ano`
10. `pct_quadra_esportes`
11. `populacao`
12. `pct_biblioteca`
13. `pct_energia_rede_publica`
14. `pct_laboratorio_informatica`

**Principais leituras:**

- **`meta_alfabetizacao` é, de longe, a variável mais influente.**
  Metas mais baixas empurram fortemente a previsão para "atingiu a
  meta"; metas mais altas empurram levemente na direção contrária.
  Isso é coerente (não é leakage, já que a meta é definida antes do
  resultado) e explica o padrão visto na EDA: estados com metas mais
  ambiciosas (ex. RS, BA) têm taxa de atingimento muito mais baixa que
  estados com metas mais brandas (ex. CE, MG, GO).
- **`pct_esgoto_rede_publica`** é a segunda variável mais relevante:
  mais cobertura de saneamento básico aumenta a chance de o município
  atingir a meta — funciona como proxy de infraestrutura/capacidade
  municipal geral.
- **Variáveis de porte do município** (`num_escolas`,
  `total_matriculas_anos_iniciais`, `total_docentes_anos_iniciais`)
  mostram um padrão de que municípios menores tendem a ter mais
  chance de atingir a meta — possivelmente por facilidade de gestão
  em escala menor, embora isso possa também refletir um efeito de
  porte do município confundido com outras variáveis (ver Limitações).
- **`pib_ultimo_disponivel`** baixo também empurra para "atingiu a
  meta", reforçando a hipótese de que as metas são calibradas em
  função do ponto de partida socioeconômico de cada município.

## 8. Insights Encontrados

- **A meta definida importa mais do que a infraestrutura do município.**
  Segundo o SHAP (seção 7), `meta_alfabetizacao` é a variável mais influente
  do modelo: municípios com metas mais baixas têm probabilidade bem maior de atingi-las, 
  enquanto metas mais ambiciosas reduzem essa chance.
  Isso sugere que parte do "risco" observado é uma função direta de como a meta foi calibrada, 
  não apenas da capacidade educacional do município.
- **Saneamento básico é a infraestrutura mais associada ao sucesso.**
  `pct_esgoto_rede_publica` foi a segunda variável mais importante. A
  cobertura de esgoto na rede pública funciona como proxy de capacidade
  administrativa e socioeconômica geral do município, não apenas de
  infraestrutura escolar.
- **Municípios menores tendem a ter mais chance de atingir a meta.**
  Variáveis de porte (`num_escolas`, `total_matriculas_anos_iniciais`,
  `total_docentes_anos_iniciais`) mostraram essa relação, possivelmente por
  facilidade de gestão em escala menor — mas o efeito pode estar confundido
  com outras variáveis (ver Limitações).
- **PIB baixo esteve associado a maior chance de atingir a meta.** Isso
  reforça a hipótese de que as metas do programa já são calibradas
  considerando o ponto de partida socioeconômico de cada município.
- **O desempenho preditivo é modesto por desenho, não por falha técnica.**
  Depois de remover corretamente as variáveis de data leakage (notas e
  proficiência da mesma avaliação), o modelo ficou com um AUC de ~0.62 —
  esperado, já que restam apenas variáveis de infraestrutura, porte e a
  meta, que têm relação real mas limitada com o resultado educacional.

## 9. Limitações do Projeto
<!-- Pessoa 2 -->

- **Granularidade:** a base é agregada por município/ano, não por
  aluno individual — o modelo prevê risco no nível municipal, não o
  resultado de um estudante específico.
- **Corte transversal:** a base final combina um único ano (2024, o
  único com meta e resultado coexistindo), o que impede análise de
  série temporal ou de estabilidade do modelo entre anos.
- **PIB setorial indisponível:** as variáveis de valor agregado por
  setor (agropecuária, indústria, serviços) vieram 100% nulas — o
  IBGE ainda não publicou a abertura setorial do PIB para o último
  ano disponível. Apenas o PIB total foi utilizado.
- **Desempenho preditivo modesto (AUC ~0.62):** esperado dado que
  todas as variáveis com forte relação direta com o resultado da
  avaliação (notas, proficiência) foram removidas por constituírem
  data leakage. As variáveis restantes (infraestrutura, porte,
  socioeconômicas) têm poder explicativo real, porém limitado.
- **Possível confusão entre porte do município e outros fatores:**
  variáveis como número de escolas e matrículas podem estar
  capturando efeito de porte do município mais do que uma relação
  causal direta com o resultado educacional.

## 10. Aplicação Prática para Políticas Públicas

- **Priorização de municípios de risco**: o modelo pode apoiar secretarias
  estaduais de educação a identificar, ainda no início do ano letivo,
  municípios com maior probabilidade de não atingir a meta, permitindo
  direcionar apoio técnico e recursos antes do resultado final.
- **Revisão da calibração de metas**: como a própria meta é o fator mais
  influente na previsão, o modelo pode ser usado para simular o impacto de
  metas alternativas antes de sua publicação, testando se uma meta é
  realista dado o perfil do município.
- **Investimento em saneamento como política educacional indireta**: a
  relação entre `pct_esgoto_rede_publica` e o atingimento da meta sugere que
  investimentos em infraestrutura básica (não apenas escolar) podem ter
  efeito colateral positivo sobre indicadores educacionais.
- **Monitoramento diferenciado por porte de município**: dado o padrão
  observado em municípios menores, políticas de alfabetização podem
  precisar de desenhos diferentes para municípios pequenos vs. grandes,
  em vez de uma meta única nacional.

## 11. Possíveis Evoluções Futuras

- Incorporar o Atlas do Desenvolvimento Humano (IDHM e indicadores
  socioeconômicos municipais) e a PNAD para variáveis de renda e
  vulnerabilidade social mais granulares.
- Investigar a abertura setorial do PIB (agropecuária, indústria, serviços)
  assim que o IBGE publicar os dados para os anos mais recentes.
- Expandir a série temporal à medida que novos anos de resultado real forem
  publicados, permitindo validação temporal do modelo (treinar em anos
  anteriores, testar em anos futuros).
- Testar modelos que separem o efeito da meta do efeito das variáveis de
  infraestrutura, para isolar melhor o que é "calibração da meta" do que é
  "capacidade real do município".
- Avaliar dados segmentados por rede Estadual, para comparar fatores de
  risco entre redes de ensino.

## Como Reproduzir

```bash
# criar ambiente
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# instalar dependências
pip install -r requirements.txt

# 1. construir a base intermediária (município, uf, regiao, ano, meta, target)
python src/preprocessing/construir_base_intermediaria.py

# 2. gerar visualizações de EDA por UF/região
python src/visualization/eda_uf_regiao_viz.py

# 3. treinar, comparar modelos, tunar hiperparâmetros e avaliar
python src/modeling/pipeline_modelagem_treino.py

# 4. gerar interpretabilidade (SHAP)
python src/evaluation/interpretabilidade_shap.py
```

## Estrutura do Repositório

```
TechChallenge-Fase3-Grupo20/
├── data/
├── notebooks/
│   ├── inspecionar_gold.py
│   └── diagnosticar_duplicatas.py
├── src/
│   ├── preprocessing/
│   │   ├── construir_target.py
│   │   └── construir_base_intermediaria.py
│   ├── modeling/
│   │   └── pipeline_modelagem_treino.py
│   ├── evaluation/
│   │   └── interpretabilidade_shap.py
│   └── visualization/
│       └── eda_uf_regiao_viz.py
├── reports/
├── images/
├── requirements.txt
├── README.md
└── .gitignore
```

## Vídeo Executivo
Link: