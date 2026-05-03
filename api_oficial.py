"""
Projeto Big Data - Dashboard de Energia Renovável no Nordeste

Sistema desenvolvido para analisar dados oficiais da ANEEL sobre energia solar
e eólica nos estados do Nordeste.

Funcionalidades:
- Ranking Nordeste
- Capacidade instalada anual
- Produção estimada mensal
- Produção estimada anual
- Projeção para 2027
- Comparação entre estados
- Dashboard online via Flask/Render

Observação:
A base da ANEEL informa capacidade instalada. A produção apresentada é uma
estimativa calculada com fator médio de capacidade.
"""

from flask import Flask, jsonify, request, send_file, render_template
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io
import os
import time
import numpy as np

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
    """
    Aqui eu carrego os dados oficiais da ANEEL.
    Usei cache para evitar baixar a base toda vez que uma consulta é feita.
    """

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
    estados = preparar_estados(estado)

    dados = dados[dados["sigufprincipal"].isin(estados)]

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
    capacidade = serie_capacidade_anual(dados)
    return capacidade * 8760 * fator


def projetar_2027(serie):
    serie = serie.dropna()

    if len(serie) < 3:
        return 0

    anos = serie.index.astype(int).values
    valores = serie.values

    coeficiente = np.polyfit(anos, valores, 1)
    previsao = np.polyval(coeficiente, 2027)

    return max(0, float(previsao))


def resumo_cards(fonte):
    df = carregar_dados()
    dados = filtrar_dados(df, "TODOS", fonte)

    total_mw = dados["potencia_kw"].sum() / 1000
    fator = fator_capacidade(fonte)
    producao_anual = total_mw * 8760 * fator

    ranking = (
        dados.groupby("sigufprincipal")["potencia_kw"]
        .sum()
        .sort_values(ascending=False) / 1000
    )

    lider = "Sem dados"
    lider_mw = 0

    if len(ranking) > 0:
        lider = nome_estado(ranking.index[0])
        lider_mw = float(ranking.iloc[0])

    total_usinas = int(len(dados))

    return {
        "total_mw": f"{total_mw:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        "producao_anual": f"{producao_anual:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        "lider": lider,
        "lider_mw": f"{lider_mw:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        "total_usinas": total_usinas
    }


def criar_grafico(tipo, estado, fonte):
    df = carregar_dados()
    titulo_fonte = fonte if fonte != "Todas" else "Solar + Eólica"

    plt.figure(figsize=(11, 6))

    if tipo == "ranking":
        base = filtrar_dados(df, "TODOS", fonte)

        ranking = (
            base.groupby("sigufprincipal")["potencia_kw"]
            .sum()
            .sort_values(ascending=False) / 1000
        )

        labels = [nome_estado(uf) for uf in ranking.index]

        plt.bar(labels, ranking.values)
        plt.title(f"Ranking Nordeste - Capacidade Instalada - {titulo_fonte}")
        plt.xlabel("Estado")
        plt.ylabel("MW instalados")
        plt.xticks(rotation=25)
        plt.grid(axis="y", linestyle="--", alpha=0.35)

    else:
        estados = preparar_estados(estado)

        for uf in estados:
            dados_estado = filtrar_dados(df, uf, fonte)

            if tipo == "capacidade_anual":
                serie = serie_capacidade_anual(dados_estado).tail(8)

                plt.plot(
                    serie.index.astype(str),
                    serie.values,
                    marker="o",
                    linewidth=2.5,
                    label=nome_estado(uf)
                )

            elif tipo == "producao_mensal":
                producao = producao_mensal_estimada(dados_estado, fonte)

                plt.plot(
                    list(producao.keys()),
                    list(producao.values()),
                    marker="o",
                    linewidth=2.5,
                    label=nome_estado(uf)
                )

            elif tipo == "producao_anual":
                serie = producao_anual_estimada(dados_estado, fonte).tail(8)

                plt.plot(
                    serie.index.astype(str),
                    serie.values,
                    marker="o",
                    linewidth=2.5,
                    label=nome_estado(uf)
                )

            elif tipo == "projecao_2027":
                serie = serie_capacidade_anual(dados_estado).tail(8)
                previsao = projetar_2027(serie)

                labels = list(serie.index.astype(str)) + ["2027"]
                valores = list(serie.values) + [previsao]

                plt.plot(
                    labels,
                    valores,
                    marker="o",
                    linewidth=2.5,
                    label=nome_estado(uf)
                )

                plt.scatter("2027", previsao, s=130)

        if tipo == "capacidade_anual":
            plt.title(f"Capacidade Instalada Anual - {titulo_fonte}")
            plt.xlabel("Ano")
            plt.ylabel("MW adicionados")

        elif tipo == "producao_mensal":
            plt.title(f"Produção Estimada Mensal - {titulo_fonte}")
            plt.xlabel("Mês")
            plt.ylabel("MWh estimados")

        elif tipo == "producao_anual":
            plt.title(f"Produção Estimada Anual - {titulo_fonte}")
            plt.xlabel("Ano")
            plt.ylabel("MWh estimados")

        elif tipo == "projecao_2027":
            plt.title(f"Projeção de Capacidade Instalada para 2027 - {titulo_fonte}")
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
        dados = filtrar_dados(df, "TODOS", fonte)

        ranking = (
            dados.groupby("sigufprincipal")["potencia_kw"]
            .sum()
            .sort_values(ascending=False) / 1000
        )

        fator = fator_capacidade(fonte)

        lider_uf = ranking.index[0]
        lider_valor = float(ranking.iloc[0])
        producao_lider = lider_valor * 8760 * fator

        texto = f"Ranking Nordeste - {titulo_fonte}\n\n"
        texto += f"O estado que lidera é {nome_estado(lider_uf)}, com {lider_valor:.2f} MW de capacidade instalada.\n"
        texto += f"A produção estimada anual do líder é de aproximadamente {producao_lider:.2f} MWh/ano.\n\n"
        texto += "Classificação dos estados:\n"

        for i, (uf, valor) in enumerate(ranking.items(), start=1):
            producao = valor * 8760 * fator
            texto += f"{i}. {nome_estado(uf)} - {valor:.2f} MW instalados - produção estimada: {producao:.2f} MWh/ano\n"

        texto += "\nObservação: a produção foi estimada com base na capacidade instalada e em fator médio de capacidade."
        return texto

    estados = preparar_estados(estado)

    if tipo == "capacidade_anual":
        texto = f"Capacidade instalada anual - {titulo_fonte}\n"
        texto += f"Estados selecionados: {nomes_estados(estados)}\n\n"

        for uf in estados:
            dados = filtrar_dados(df, uf, fonte)
            total = dados["potencia_kw"].sum() / 1000
            serie = serie_capacidade_anual(dados).tail(8)

            texto += f"{nome_estado(uf)}\n"
            texto += f"Capacidade total instalada: {total:.2f} MW\n"
            texto += "Histórico anual:\n"

            for ano, valor in serie.items():
                texto += f"{ano}: {valor:.2f} MW adicionados\n"

            texto += "\n"

        return texto

    if tipo == "producao_mensal":
        texto = f"Produção estimada mensal - {titulo_fonte}\n"
        texto += f"Estados selecionados: {nomes_estados(estados)}\n\n"

        for uf in estados:
            dados = filtrar_dados(df, uf, fonte)
            producao = producao_mensal_estimada(dados, fonte)

            texto += f"{nome_estado(uf)}\n"

            for mes, valor in producao.items():
                texto += f"{mes}: {valor:.2f} MWh estimados\n"

            texto += "\n"

        texto += "Observação: valores estimados a partir da capacidade instalada."
        return texto

    if tipo == "producao_anual":
        texto = f"Produção estimada anual - {titulo_fonte}\n"
        texto += f"Estados selecionados: {nomes_estados(estados)}\n\n"

        for uf in estados:
            dados = filtrar_dados(df, uf, fonte)
            serie = producao_anual_estimada(dados, fonte).tail(8)

            texto += f"{nome_estado(uf)}\n"

            for ano, valor in serie.items():
                texto += f"{ano}: {valor:.2f} MWh estimados\n"

            texto += "\n"

        texto += "Observação: produção calculada com horas anuais e fator médio de capacidade."
        return texto

    if tipo == "projecao_2027":
        texto = f"Projeção de capacidade instalada para 2027 - {titulo_fonte}\n"
        texto += f"Estados selecionados: {nomes_estados(estados)}\n\n"

        for uf in estados:
            dados = filtrar_dados(df, uf, fonte)
            serie = serie_capacidade_anual(dados).tail(8)
            previsao = projetar_2027(serie)

            ultimo_ano = int(serie.index[-1]) if len(serie) > 0 else 2026
            ultimo_valor = float(serie.iloc[-1]) if len(serie) > 0 else 0

            crescimento = 0
            if ultimo_valor != 0:
                crescimento = ((previsao - ultimo_valor) / ultimo_valor) * 100

            texto += f"{nome_estado(uf)}\n"
            texto += "Histórico recente:\n"

            for ano, valor in serie.items():
                texto += f"{ano}: {valor:.2f} MW adicionados\n"

            texto += f"2027 projetado: {previsao:.2f} MW\n"
            texto += f"Crescimento estimado em relação a {ultimo_ano}: {crescimento:.2f}%\n\n"

        texto += "Observação: projeção calculada por tendência linear simples."
        return texto

    return "Não foi possível reconhecer o tipo de análise selecionado."


@app.route("/api/resumo")
def api_resumo():
    fonte = request.args.get("fonte", "Todas")
    return jsonify(resumo_cards(fonte))


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