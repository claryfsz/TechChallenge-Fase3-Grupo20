## Construção da base intermediária (município, uf, regiao, ano, meta, target)


import unicodedata
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

PATH_INDICADOR = "data/gold_indicador_municipio.csv"
PATH_META = "data/gold_meta_vs_resultado.csv"

# Mapeamento UF -> Região (não depende do target, então não é leakage)
REGIAO_POR_UF = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte",
    "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}


def normalizar_texto(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")


def construir_base():
    indicador = pd.read_csv(PATH_INDICADOR)
    meta = pd.read_csv(PATH_META)

    meta["target"] = (meta["diferenca"] >= 0).astype(int)

    for df in (indicador, meta):
        df["municipio_norm"] = df["municipio"].apply(normalizar_texto)
        df["uf_norm"] = df["uf"].astype(str).str.strip().str.upper()

    indicador_agg = (
        indicador.groupby(["id_municipio", "municipio_norm", "uf_norm", "ano"])
        .agg(indicador=("indicador", "mean"))
        .reset_index()
    )

    base = meta.merge(
        indicador_agg,
        on=["municipio_norm", "uf_norm", "ano"],
        how="left",
    )

    base["regiao"] = base["uf_norm"].map(REGIAO_POR_UF)

    base_final = base[
        ["municipio", "uf_norm", "regiao", "ano", "meta", "target"]
    ].rename(columns={"uf_norm": "uf"})

    return base_final


def main():
    base = construir_base()
    print(f"Base intermediária: {base.shape[0]} linhas x {base.shape[1]} colunas")
    base.to_csv("data/base_intermediaria.csv", index=False)
    print("Salvo em data/base_intermediaria.csv")
    return base


if __name__ == "__main__":
    main()