## Diagnóstico: por que o merge duplicou linhas?
## Investiga se há município+uf+ano repetido em indicador_municipio.

import unicodedata
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

indicador = pd.read_csv("data/gold_indicador_municipio.csv")


def normalizar_texto(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")


indicador["municipio_norm"] = indicador["municipio"].apply(normalizar_texto)
indicador["uf_norm"] = indicador["uf"].astype(str).str.strip().str.upper()

# Conta quantas vezes cada combinação município+uf+ano aparece
contagem = (
    indicador.groupby(["municipio_norm", "uf_norm", "ano"])
    .size()
    .reset_index(name="qtd_linhas")
)

duplicados = contagem[contagem["qtd_linhas"] > 1].sort_values(
    "qtd_linhas", ascending=False
)

print(f"Total de combinações município+uf+ano únicas: {len(contagem)}")
print(f"Combinações com mais de 1 linha: {len(duplicados)}")

if len(duplicados) > 0:
    print("\nTop combinações duplicadas:")
    print(duplicados.head(10))

    # Mostra o detalhe de um caso duplicado pra entender o que muda entre as linhas
    exemplo = duplicados.iloc[0]
    print(f"\nDetalhe do exemplo: {exemplo['municipio_norm']} / "
          f"{exemplo['uf_norm']} / {exemplo['ano']}")
    print(
        indicador[
            (indicador["municipio_norm"] == exemplo["municipio_norm"])
            & (indicador["uf_norm"] == exemplo["uf_norm"])
            & (indicador["ano"] == exemplo["ano"])
        ]
    )

# Também mostra quais anos existem na base (pode ser que o indicador tenha
# mais anos que meta_vs_resultado, o que não causaria duplicação sozinho,
# mas ajuda a entender a estrutura)
print("\nAnos presentes em indicador_municipio:", sorted(indicador["ano"].unique()))