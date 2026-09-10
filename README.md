# Tech Challenge – Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil

## 1. Contexto do Problema
<!-- Pessoa 1 --> 
Breve descrição do problema educacional: por que a alfabetização infantil é um
indicador crítico, e por que antecipar riscos importa para gestores públicos.

## 2. Objetivo Analítico
<!-- Pessoa 1 --> 
Objetivo do projeto: desenvolver um modelo supervisionado para prever se um
aluno será considerado alfabetizado ou não, a partir de variáveis
educacionais, territoriais e socioeconômicas.

## 3. Descrição da Base Utilizada
<!-- Pessoa 1 --> 
- Origem: camada Gold construída na Fase 2
- Principais grupos de variáveis (indicadores de alfabetização, metas,
  dados territoriais, socioeconômicos, educacionais complementares,
  populacionais)
- Fontes externas usadas para enriquecimento (se houver): IBGE, Censo
  Escolar, FUNDEB, PNAD, Atlas do Desenvolvimento Humano

## 4. Etapas de Modelagem
<!-- Pessoa 2 --> 
- Tratamento de valores faltantes (estratégia de imputação por tipo de
  variável)
- Transformação de variáveis numéricas (escala) e categóricas (encoding)
- Tratamento de data leakage: quais variáveis foram removidas e por quê
- Split treino/validação/teste (estratégia e proporções)
- Pipeline Scikit-learn (pré-processamento integrado ao modelo)

## 5. Escolha do Algoritmo
<!-- Pessoa 2 --> 
- Modelos testados (ex.: Regressão Logística, Random Forest, Gradient
  Boosting) e critério de comparação
- Estratégia de validação cruzada e ajuste de hiperparâmetros
- Justificativa da escolha do modelo final (performance +
  interpretabilidade + objetivo de negócio)

## 6. Métricas de Avaliação
<!-- Pessoa 2 --> 
- Acurácia, precisão, recall, F1, AUC (treino x teste)
- Análise de overfitting (gap treino/teste)
- Matriz de confusão

## 7. Interpretação dos Resultados
<!-- Pessoa 2 --> 
- Feature Importance / SHAP Values
- Quais variáveis mais influenciam a predição

## 8. Insights Encontrados
<!-- Pessoa 1 + Pessoa 2 --> 
- Principais achados da EDA e da modelagem, traduzidos para linguagem
  acessível a gestores públicos

## 9. Limitações do Projeto
<!-- Pessoa 2 --> 
- Limitações dos dados, do modelo, e do escopo da análise

## 10. Aplicação Prática para Políticas Públicas
<!-- Pessoa 1 --> 
- Como os resultados podem apoiar decisões (identificação de municípios
  de risco, priorização de recursos, etc.)

## 11. Possíveis Evoluções Futuras
<!-- Pessoa 1 + Pessoa 2 --> 
- Próximos passos: novas fontes de dados, outros modelos, monitoramento
  contínuo, etc.

## Como Reproduzir

```bash
# criar ambiente
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# instalar dependências
pip install -r requirements.txt

# rodar pipeline
python src/modeling/train.py
```

## Estrutura do Repositório

```
tech-challenge-fase3/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   ├── evaluation/
│   └── visualization/
├── reports/
├── images/
├── requirements.txt
├── README.md
└── .gitignore
```

## Vídeo Executivo
Link: <inserir link do vídeo>