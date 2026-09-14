## Visualizações: taxa de municípios que atingiram a meta, por UF e por região

## Requer que data/base_intermediaria.csv já exista
# - (gerado por src/preprocessing/construir_base_intermediaria.py).

## Gera:
# - images/taxa_atingimento_por_uf.png
# - images/taxa_atingimento_por_regiao.png

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)


def eda_taxa_por_grupo(base: pd.DataFrame, coluna_grupo: str, titulo: str, arquivo: str):
    taxa = (
        base.groupby(coluna_grupo)["target"]
        .mean()
        .sort_values(ascending=False)
    )

    print(f"\nTaxa de municípios que atingiram a meta, por {coluna_grupo}:")
    print(taxa.round(3))

    plt.figure(figsize=(10, 5))
    taxa.plot(kind="bar", color="#2E86AB")
    plt.title(titulo)
    plt.ylabel("Taxa de municípios que atingiram a meta")
    plt.xlabel(coluna_grupo.upper())
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)
    plt.close()
    print(f"Gráfico salvo em: {arquivo}")


def main():
    base = pd.read_csv("data/base_intermediaria.csv")

    eda_taxa_por_grupo(
        base, "regiao",
        "Taxa de municípios que atingiram a meta, por região",
        "images/taxa_atingimento_por_regiao.png",
    )

    eda_taxa_por_grupo(
        base, "uf",
        "Taxa de municípios que atingiram a meta, por UF",
        "images/taxa_atingimento_por_uf.png",
    )

    print("\nEstatísticas da coluna 'meta' por ano:")
    print(base.groupby("ano")["meta"].describe())


if __name__ == "__main__":
    main()