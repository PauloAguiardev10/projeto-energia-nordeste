"""
Projeto de Big Data - Energia Renovável no Nordeste

Neste projeto eu utilizei dados oficiais da ANEEL para analisar
a capacidade instalada de energia solar e eólica nos estados do Nordeste.

Também implementei:
- Ranking dos estados
- Geração de gráfico
- Interface web acessível via navegador (computador e celular)
"""

from flask import Flask, jsonify, send_file, render_template
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import time
import os

app = Flask(__name__)

# Base oficial da ANEEL
URL_SIGA = "https://dadosabertos.aneel.gov.br/dataset/6d90b77c-c5f5-4d81-bdec-7bc619494bb9/resource/2f65a1b0-19b8-4360-8238-b34ab4693d55/download/siga-empreendimentos-geracao-diario.csv"

# Estados do Nordeste
ESTADOS = {
    "AL": "Alagoas",
    "BA": "Bahia",
    "CE": "Ceará",
    "MA": "Maranhão",
    "PB": "Paraíba",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RN": "Rio Grande do Norte",
    "SE": "Sergipe"
}

# Cache para não baixar dados toda hora
CACHE = {"df": None, "hora": 0}


@app.route("/")
def home():
    return render_template("index.html")


def carregar_dados():
    """
    Aqui eu carrego os dados da ANEEL e faço um tratamento básico.
    Também uso cache para melhorar a performance.
    """

    agora = time.time()

    if CACHE["df"] is not None and agora - CACHE["hora"] < 1800:
        return CACHE["df"]

    df = pd.read_csv(URL_SIGA, sep=";", encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filtra apenas Nordeste
    df = df[df["sigufprincipal"].isin(ESTADOS.keys())]

    # Converte potência
    df["potencia_kw"] = pd.to_numeric(
        df["mdapotenciafiscalizadakw"], errors="coerce"
    ).fillna(0)

    # Identifica fonte
    df["fonte"] = df["nomfontecombustivel"].astype(str).str.lower()

    def classificar(f):
        if "solar" in f:
            return "Solar"
        if "eol" in f:
            return "Eólica"
        return "Outros"

    df["fonte_padronizada"] = df["fonte"].apply(classificar)

    df = df[df["fonte_padronizada"].isin(["Solar", "Eólica"])]

    CACHE["df"] = df
    CACHE["hora"] = agora

    return df


@app.route("/api/analise")
def analise():
    """
    Aqui eu gero o ranking dos estados com base na capacidade instalada.
    """

    df = carregar_dados()

    ranking = (
        df.groupby("sigufprincipal")["potencia_kw"]
        .sum()
        .sort_values(ascending=False) / 1000
    )

    texto = "⚡ Ranking de Energia Renovável no Nordeste\n\n"

    for i, (uf, valor) in enumerate(ranking.items(), 1):
        texto += f"{i}. {ESTADOS[uf]} - {valor:.2f} MW instalados\n"

    texto += "\nFonte: Dados oficiais da ANEEL."

    return jsonify({
        "resposta": texto,
        "grafico_url": "/api/grafico"
    })


@app.route("/api/grafico")
def grafico():
    """
    Aqui eu gero o gráfico do ranking.
    """

    df = carregar_dados()

    ranking = (
        df.groupby("sigufprincipal")["potencia_kw"]
        .sum()
        .sort_values(ascending=False) / 1000
    )

    plt.figure(figsize=(10, 5))
    plt.bar([ESTADOS[x] for x in ranking.index], ranking.values)

    plt.title("Ranking de Energia Renovável - Nordeste")
    plt.xticks(rotation=30)
    plt.ylabel("MW Instalados")

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)

    return send_file(img, mimetype="image/png")


if __name__ == "__main__":
    print("Servidor rodando...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))