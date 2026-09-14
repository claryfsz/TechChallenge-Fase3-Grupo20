## Construção do target + teste de merge entre as tabelas Gold

## IMPORTANTE (data leakage):
# - 'resultado' e 'diferenca' NUNCA podem virar features do modelo -- elas
#  são a origem do target. Só 'meta' pode eventualmente virar feature,
#  porque é definida antes do resultado sair.

## IMPORTANTE (qualidade dos dados):
# - indicador_municipio tem município+uf+ano duplicado em ~2.100 casos
# (o mesmo município/ano aparece com valores de indicador diferentes, provavelmente por rede de ensino, série ou edição do Saeb não agregada). 
# Este script agrega pela MÉDIA antes de qualquer merge.

import unicodedata
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

PATH_INDICADOR = "data/gold_indicador_municipio.csv"
PATH_META = "data/gold_meta_vs_resultado.csv"


def normalizar_texto(s: str) -> str:
    """Remove acentos, espaços extras e padroniza caixa, pra facilitar o merge."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s


def main():
    indicador = pd.read_csv(PATH_INDICADOR)
    meta = pd.read_csv(PATH_META)

    # -----------------------------------------------------------------
    # 1. Criar o target a partir de meta_vs_resultado
    # -----------------------------------------------------------------
    meta["target"] = (meta["diferenca"] >= 0).astype(int)

    print("Distribuição do target (1 = atingiu a meta):")
    print(meta["target"].value_counts(normalize=True).round(3))

    # -----------------------------------------------------------------
    # 2. Normalizar chaves de junção nas duas tabelas
    # -----------------------------------------------------------------
    for df in (indicador, meta):
        df["municipio_norm"] = df["municipio"].apply(normalizar_texto)
        df["uf_norm"] = df["uf"].astype(str).str.strip().str.upper()

    # -----------------------------------------------------------------
    # 2b. Agregar duplicatas de município+uf+ano em indicador_municipio
    #     (ver nota de qualidade de dados no topo do arquivo)
    # -----------------------------------------------------------------
    linhas_antes = len(indicador)
    indicador = (
        indicador.groupby(["id_municipio", "municipio_norm", "uf_norm", "ano"])
        .agg(indicador=("indicador", "mean"))
        .reset_index()
    )
    print(f"\nindicador_municipio: {linhas_antes} linhas antes da agregação, "
          f"{len(indicador)} depois (agregado por média em município+uf+ano).")

    # -----------------------------------------------------------------
    # 3. Merge por município (normalizado) + UF + ano
    # -----------------------------------------------------------------
    base = meta.merge(
        indicador,
        on=["municipio_norm", "uf_norm", "ano"],
        how="left",
        suffixes=("_meta", "_indicador"),
    )

    total = len(meta)
    sem_match = base["id_municipio"].isna().sum()

    print(f"\nTotal de linhas em meta_vs_resultado: {total}")
    print(f"Linhas sem correspondência em indicador_municipio: {sem_match} "
          f"({sem_match / total:.1%})")

    if sem_match > 0:
        print("\nExemplos de município/UF/ano que não bateram:")
        print(
            base.loc[base["id_municipio"].isna(), ["municipio_meta", "uf_norm", "ano"]]
            .drop_duplicates()
            .head(15)
        )
        print(
            "\n[Ação sugerida] Confira se são nomes de município grafados de forma "
            "diferente entre as bases (hífen, abreviação, etc.), ano fora do range "
            "de uma das tabelas, ou município realmente ausente em uma delas."
        )

    # -----------------------------------------------------------------
    # 4. Conferir se 'resultado' e 'indicador' realmente batem
    #    (esperado: devem ser a mesma coisa, ou muito próximos)
    # -----------------------------------------------------------------
    comparaveis = base.dropna(subset=["resultado", "indicador"])
    if len(comparaveis) > 0:
        diff_media = (comparaveis["resultado"] - comparaveis["indicador"]).abs().mean()
        print(f"\nDiferença média absoluta entre 'resultado' e 'indicador' "
              f"(deveriam ser a mesma métrica): {diff_media:.4f}")

    return base


if __name__ == "__main__":
    base_unificada = main()
    print(f"\nBase unificada final: {base_unificada.shape[0]} linhas x "
          f"{base_unificada.shape[1]} colunas")
    print("\nColunas disponíveis:", list(base_unificada.columns))