"""
Projeto de Big Data - Energia Renovável no Nordeste

Este sistema foi desenvolvido para analisar dados oficiais da ANEEL sobre
energia solar e eólica nos estados do Nordeste.

O projeto trabalha com:
- Capacidade instalada anual
- Produção estimada mensal
- Produção estimada anual
- Projeção anual para 2027
- Ranking Nordeste
- Comparação entre estados

Observação:
A base da ANEEL informa principalmente capacidade instalada. Por isso,
a produção apresentada no projeto é uma estimativa calculada com base
na capacidade instalada e em fatores médios de capacidade.
"""

from flask import Flask, jsonify, request, send_file, render_template
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io
import time
import numpy as np
import os

app = Flask(__name__)

URL_SIGA = "https://dadosabertos.aneel.gov.br/dataset/6d90b77c-c5f5-4d81-bdec-7bc619494bb9/resource/2f65a1b0-19b8-4360-8238-b34ab4693d55/download/siga-empreendimentos-geracao-diario.csv"

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

MESES = {
    1: "Jan",
    2: "Fev",
    3: "Mar",
    4: "Abr",
    5: "Mai",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Set",
    10: "Out",
    11: "Nov",
    12: "Dez"
}

CACHE = {
    "df": None,
    "hora": 0
}


@app.after_request
def permitir_frontend(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def home():
    return render_template("index.html")


def nome_estado(uf):
    return ESTADOS.get(uf, uf)


def nomes_estados(lista_ufs):
    return ", ".join([nome_estado(uf) for uf in lista_ufs])


def fator_capacidade(fonte):
    if fonte == "Solar":
        return 0.22

    if fonte == "Eólica":
        return 0.45

    return 0.33


def carregar_dados():
    agora = time.time()

    if CACHE["df"] is not None and agora - CACHE["hora"] < 1800:
        return CACHE["df"].copy()

    df = pd.read_csv(URL_SIGA, sep=";", encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    df = df[df["sigufprincipal"].isin(ESTADOS.keys())].copy()

    df["potencia_kw"] = pd.to_numeric(
        df["mdapotenciafiscalizadakw"],
        errors="coerce"
    ).fillna(0)

    df["data_operacao"] = pd.to_datetime(
        df["datentradaoperacao"],
        errors="coerce",
        format="mixed"
    )

    df["fonte_original"] = df["nomfontecombustivel"].astype(str).str.lower()
    df["tipo_original"] = df["sigtipogeracao"].astype(str).str.lower()

    def classificar_fonte(linha):
        fonte = linha["fonte_original"]
        tipo = linha["tipo_original"]

        if "solar" in fonte or "fotovolta" in fonte or "ufv" in tipo:
            return "Solar"

        if "eol" in fonte or "vento" in fonte:
            return "Eólica"

        return "Outros"

    df["fonte_padronizada"] = df.apply(classificar_fonte, axis=1)
    df = df[df["fonte_padronizada"].isin(["Solar", "Eólica"])].copy()

    CACHE["df"] = df.copy()
    CACHE["hora"] = agora

    return df.copy()


def preparar_estados(estado):
    if not estado or estado == "TODOS":
        return list(ESTADOS.keys())

    return [uf.strip() for uf in estado.split(",") if uf.strip() in ESTADOS]


def filtrar_dados(df, estado, fonte):
    dados = df.copy()
    lista_estados = preparar_estados(estado)

    dados = dados[dados["sigufprincipal"].isin(lista_estados)]

    if fonte and fonte != "Todas":
        dados = dados[dados["fonte_padronizada"] == fonte]

    return dados


def serie_capacidade_anual(dados):
    dados = dados.dropna(subset=["data_operacao"]).copy()
    dados["ano"] = dados["data_operacao"].dt.year

    return (
        dados.groupby("ano")["potencia_kw"]
        .sum()
        .sort_index() / 1000
    )


def producao_mensal_estimada(dados, fonte):
    fator = fator_capacidade(fonte)

    dados = dados.dropna(subset=["data_operacao"]).copy()
    dados["mes_numero"] = dados["data_operacao"].dt.month

    capacidade_mw = (
        dados.groupby("mes_numero")["potencia_kw"]
        .sum()
        .sort_index() / 1000
    )

    producao_mwh = capacidade_mw * 730 * fator

    resultado = {}

    for mes in range(1, 13):
        resultado[MESES[mes]] = float(producao_mwh.get(mes, 0))

    return resultado


def producao_anual_estimada(dados, fonte):
    fator = fator_capacidade(fonte)
    capacidade_anual = serie_capacidade_anual(dados)
    producao_mwh = capacidade_anual * 8760 * fator

    return producao_mwh


def projetar_para_2027(serie):
    serie = serie.dropna()

    if len(serie) < 3:
        return 0

    anos = serie.index.astype(int).values
    valores = serie.values

    coeficiente = np.polyfit(anos, valores, 1)
    previsao_2027 = np.polyval(coeficiente, 2027)

    return max(0, float(previsao_2027))


def criar_grafico(tipo, estado, fonte):
    df = carregar_dados()
    titulo_fonte = fonte if fonte != "Todas" else "Solar + Eólica"

    plt.figure(figsize=(11, 6))

    if tipo == "ranking":
        base = df.copy()

        if fonte != "Todas":
            base = base[base["fonte_padronizada"] == fonte]

        ranking = (
            base.groupby("sigufprincipal")["potencia_kw"]
            .sum()
            .sort_values(ascending=False) / 1000
        )

        labels = [nome_estado(uf) for uf in ranking.index]

        plt.bar(labels, ranking.values)
        plt.title(f"Ranking Nordeste - Capacidade Instalada - {titulo_fonte}")
        plt.xlabel("Estado")
        plt.ylabel("Capacidade instalada em MW")
        plt.xticks(rotation=25)
        plt.grid(axis="y", linestyle="--", alpha=0.35)

    else:
        lista_estados = preparar_estados(estado)

        for uf in lista_estados:
            dados_estado = filtrar_dados(df, uf, fonte)

            if tipo == "capacidade_anual":
                serie = serie_capacidade_anual(dados_estado).tail(8)

                plt.plot(
                    serie.index.astype(str),
                    serie.values,
                    marker="o",
                    linewidth=2,
                    label=nome_estado(uf)
                )

            elif tipo == "producao_mensal":
                producao = producao_mensal_estimada(dados_estado, fonte)
                meses = list(producao.keys())
                valores = list(producao.values())

                plt.plot(
                    meses,
                    valores,
                    marker="o",
                    linewidth=2,
                    label=nome_estado(uf)
                )

            elif tipo == "producao_anual":
                serie = producao_anual_estimada(dados_estado, fonte).tail(8)

                plt.plot(
                    serie.index.astype(str),
                    serie.values,
                    marker="o",
                    linewidth=2,
                    label=nome_estado(uf)
                )

            elif tipo == "projecao_2027":
                serie = serie_capacidade_anual(dados_estado).tail(8)
                previsao = projetar_para_2027(serie)

                labels = list(serie.index.astype(str)) + ["2027"]
                valores = list(serie.values) + [previsao]

                plt.plot(
                    labels,
                    valores,
                    marker="o",
                    linewidth=2,
                    label=nome_estado(uf)
                )

                plt.scatter("2027", previsao, s=120)

        if tipo == "capacidade_anual":
            plt.title(f"Capacidade instalada anual - {titulo_fonte}")
            plt.xlabel("Ano")
            plt.ylabel("MW adicionados no ano")

        elif tipo == "producao_mensal":
            plt.title(f"Produção estimada mensal - {titulo_fonte}")
            plt.xlabel("Meses do ano")
            plt.ylabel("Produção estimada em MWh")

        elif tipo == "producao_anual":
            plt.title(f"Produção estimada anual - {titulo_fonte}")
            plt.xlabel("Ano")
            plt.ylabel("Produção estimada em MWh")

        elif tipo == "projecao_2027":
            plt.title(f"Projeção de capacidade instalada para 2027 - {titulo_fonte}")
            plt.xlabel("Ano")
            plt.ylabel("MW adicionados/projetados")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()

    imagem = io.BytesIO()
    plt.savefig(imagem, format="png", dpi=130)
    plt.close()
    imagem.seek(0)

    return imagem


def montar_resposta(tipo, estado, fonte):
    df = carregar_dados()
    titulo_fonte = fonte if fonte != "Todas" else "Solar + Eólica"

    if tipo == "ranking":
        base = df.copy()

        if fonte != "Todas":
            base = base[base["fonte_padronizada"] == fonte]

        ranking = (
            base.groupby("sigufprincipal")["potencia_kw"]
            .sum()
            .sort_values(ascending=False) / 1000
        )

        fator = fator_capacidade(fonte)

        lider_uf = ranking.index[0]
        lider_valor = ranking.iloc[0]
        producao_lider = lider_valor * 8760 * fator

        texto = f"Ranking Nordeste - {titulo_fonte}\n\n"
        texto += f"O estado que lidera é {nome_estado(lider_uf)}, com {lider_valor:.2f} MW de capacidade instalada.\n"
        texto += f"A produção estimada anual desse estado é de aproximadamente {producao_lider:.2f} MWh/ano.\n\n"
        texto += "Classificação dos estados:\n"

        for i, (uf, valor) in enumerate(ranking.items(), start=1):
            producao_estimada = valor * 8760 * fator

            texto += (
                f"{i}. {nome_estado(uf)} - "
                f"{valor:.2f} MW instalados - "
                f"produção estimada: {producao_estimada:.2f} MWh/ano\n"
            )

        texto += "\nObservação: a produção foi estimada com base na capacidade instalada e em fator médio de capacidade."

        return texto

    lista_estados = preparar_estados(estado)
    texto = ""

    if tipo == "capacidade_anual":
        texto += f"Capacidade instalada anual - {titulo_fonte}\n"
        texto += f"Estados selecionados: {nomes_estados(lista_estados)}\n\n"

        for uf in lista_estados:
            dados_estado = filtrar_dados(df, uf, fonte)
            total_mw = dados_estado["potencia_kw"].sum() / 1000
            serie = serie_capacidade_anual(dados_estado).tail(8)

            texto += f"{nome_estado(uf)}\n"
            texto += f"Capacidade total instalada encontrada: {total_mw:.2f} MW\n"
            texto += "Últimos anos analisados:\n"

            for ano, valor in serie.items():
                texto += f"{ano}: {valor:.2f} MW adicionados\n"

            texto += "\n"

        return texto

    if tipo == "producao_mensal":
        texto += f"Produção estimada mensal - {titulo_fonte}\n"
        texto += f"Estados selecionados: {nomes_estados(lista_estados)}\n\n"

        for uf in lista_estados:
            dados_estado = filtrar_dados(df, uf, fonte)
            producao = producao_mensal_estimada(dados_estado, fonte)

            texto += f"{nome_estado(uf)}\n"

            for mes, valor in producao.items():
                texto += f"{mes}: {valor:.2f} MWh estimados\n"

            texto += "\n"

        texto += "Observação: a produção mensal foi estimada a partir da capacidade instalada."
        return texto

    if tipo == "producao_anual":
        texto += f"Produção estimada anual - {titulo_fonte}\n"
        texto += f"Estados selecionados: {nomes_estados(lista_estados)}\n\n"

        for uf in lista_estados:
            dados_estado = filtrar_dados(df, uf, fonte)
            serie = producao_anual_estimada(dados_estado, fonte).tail(8)

            texto += f"{nome_estado(uf)}\n"

            for ano, valor in serie.items():
                texto += f"{ano}: {valor:.2f} MWh estimados\n"

            texto += "\n"

        texto += "Observação: a produção anual foi estimada usando horas anuais e fator médio de capacidade."
        return texto

    if tipo == "projecao_2027":
        texto += f"Projeção de capacidade instalada para 2027 - {titulo_fonte}\n"
        texto += f"Estados selecionados: {nomes_estados(lista_estados)}\n\n"

        for uf in lista_estados:
            dados_estado = filtrar_dados(df, uf, fonte)
            serie = serie_capacidade_anual(dados_estado).tail(8)
            previsao_2027 = projetar_para_2027(serie)

            ultimo_ano = int(serie.index[-1]) if len(serie) > 0 else 2026
            ultimo_valor = float(serie.iloc[-1]) if len(serie) > 0 else 0

            crescimento = 0
            if ultimo_valor != 0:
                crescimento = ((previsao_2027 - ultimo_valor) / ultimo_valor) * 100

            texto += f"{nome_estado(uf)}\n"
            texto += "Histórico recente:\n"

            for ano, valor in serie.items():
                texto += f"{ano}: {valor:.2f} MW adicionados\n"

            texto += f"2027 projetado: {previsao_2027:.2f} MW\n"
            texto += f"Crescimento estimado em relação a {ultimo_ano}: {crescimento:.2f}%\n\n"

        texto += "Observação: a projeção para 2027 foi calculada por tendência linear simples, usando os anos anteriores como base."
        return texto

    return "Não foi possível reconhecer o tipo de análise selecionado."


@app.route("/api/analise")
def api_analise():
    tipo = request.args.get("tipo", "ranking")
    estado = request.args.get("estado", "TODOS")
    fonte = request.args.get("fonte", "Todas")

    resposta = montar_resposta(tipo, estado, fonte)

    return jsonify({
        "resposta": resposta,
        "grafico_url": f"/api/grafico?tipo={tipo}&estado={estado}&fonte={fonte}"
    })


@app.route("/api/grafico")
def api_grafico():
    tipo = request.args.get("tipo", "ranking")
    estado = request.args.get("estado", "TODOS")
    fonte = request.args.get("fonte", "Todas")

    imagem = criar_grafico(tipo, estado, fonte)

    return send_file(imagem, mimetype="image/png")


if __name__ == "__main__":
    print("Servidor iniciado com sucesso.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))