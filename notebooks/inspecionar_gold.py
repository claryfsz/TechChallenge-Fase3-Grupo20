## Carregamento e inspeção inicial da base Gold (Fase 2)

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

CAMINHOS = {
    "indicador_municipio": "data/gold_indicador_municipio.csv",
    "meta_vs_resultado": "data/gold_meta_vs_resultado.csv",
    "evolucao_temporal": "data/gold_evolucao_temporal.csv",
}


def inspecionar(nome, caminho):
    print(f"\n{'=' * 70}")
    print(f"  {nome}  ({caminho})")
    print("=" * 70)

    df = pd.read_csv(caminho)

    print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")

    print("\nColunas e tipos:")
    print(df.dtypes)

    print("\nValores nulos por coluna:")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0] if nulos.sum() > 0 else "  (nenhum nulo encontrado)")

    print("\nDuplicados:", df.duplicated().sum())

    print("\nPrimeiras linhas:")
    print(df.head())

    return df


if __name__ == "__main__":
    dataframes = {}
    for nome, caminho in CAMINHOS.items():
        try:
            dataframes[nome] = inspecionar(nome, caminho)
        except FileNotFoundError:
            print(f"\n[AVISO] Arquivo não encontrado: {caminho}")
            print("Baixe o CSV da pasta gold/csv do repositório da Fase 2 e "
                  "coloque em data/ com esse nome.")

