from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import os
import time

app = Flask(__name__)

URL_SIGA = "https://dadosabertos.aneel.gov.br/dataset/6d90b77c-c5f5-4d81-bdec-7bc619494bb9/resource/2f65a1b0-19b8-4360-8238-b34ab4693d55/download/siga-empreendimentos-geracao-diario.csv"

ESTADOS = {
    "AL": "Alagoas", "BA": "Bahia", "CE": "Ceará", "MA": "Maranhão",
    "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí",
    "RN": "Rio Grande do Norte", "SE": "Sergipe"
}

MESES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}

CACHE = {"df": None, "hora": 0}


@app.route("/")
def home():
    return render_template("index.html")


def nome_estado(uf):
    return ESTADOS.get(uf, uf)


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

    estados = [uf.strip() for uf in estado.split(",") if uf.strip() in ESTADOS]
    return estados if estados else ["CE"]


def filtrar(df, estado, fonte):
    estados = preparar_estados(estado)
    dados = df[df["sigufprincipal"].isin(estados)].copy()

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


def producao_anual(dados, fonte):
    return serie_capacidade_anual(dados) * 8760 * fator_capacidade(fonte)


def producao_mensal(dados, fonte):
    fator = fator_capacidade(fonte)

    dados = dados.dropna(subset=["data_operacao"]).copy()
    dados["mes"] = dados["data_operacao"].dt.month

    capacidade = (
        dados.groupby("mes")["potencia_kw"]
        .sum()
        .sort_index() / 1000
    )

    producao = capacidade * 730 * fator
    return {MESES[m]: float(producao.get(m, 0)) for m in range(1, 13)}


def projetar_2027(serie):
    serie = serie.dropna()

    if len(serie) < 3:
        return 0

    anos = serie.index.astype(int).values
    valores = serie.values

    coef = np.polyfit(anos, valores, 1)
    previsao = np.polyval(coef, 2027)

    return max(0, float(previsao))


@app.route("/api/resumo")
def api_resumo():
    fonte = request.args.get("fonte", "Todas")

    df = carregar_dados()
    dados = filtrar(df, "TODOS", fonte)

    total_mw = dados["potencia_kw"].sum() / 1000
    producao_estimada = total_mw * 8760 * fator_capacidade(fonte)

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

    return jsonify({
        "total_mw": round(float(total_mw), 2),
        "producao_anual": round(float(producao_estimada), 2),
        "lider": lider,
        "lider_mw": round(lider_mw, 2),
        "registros": int(len(dados))
    })


@app.route("/api/comparativo")
def api_comparativo():
    estado = request.args.get("estado", "CE")
    fonte = request.args.get("fonte", "Todas")

    df = carregar_dados()
    estados = preparar_estados(estado)

    cards = []

    for uf in estados:
        dados = filtrar(df, uf, fonte)
        total_mw = dados["potencia_kw"].sum() / 1000
        producao_estimada = total_mw * 8760 * fator_capacidade(fonte)

        cards.append({
            "uf": uf,
            "estado": nome_estado(uf),
            "capacidade_mw": round(float(total_mw), 2),
            "producao_mwh": round(float(producao_estimada), 2),
            "registros": int(len(dados))
        })

    return jsonify(cards)


@app.route("/api/dados_grafico")
def api_dados_grafico():
    tipo = request.args.get("tipo", "ranking")
    estado = request.args.get("estado", "TODOS")
    fonte = request.args.get("fonte", "Todas")

    df = carregar_dados()

    if tipo == "ranking":
        dados = filtrar(df, "TODOS", fonte)

        ranking = (
            dados.groupby("sigufprincipal")["potencia_kw"]
            .sum()
            .sort_values(ascending=False) / 1000
        )

        return jsonify({
            "tipo_grafico": "bar",
            "labels": [nome_estado(uf) for uf in ranking.index],
            "datasets": [{
                "label": "Capacidade instalada (MW)",
                "data": [round(float(v), 2) for v in ranking.values]
            }]
        })

    estados = preparar_estados(estado)
    datasets = []

    labels_finais = []

    for uf in estados:
        dados_estado = filtrar(df, uf, fonte)

        if tipo == "expansao_usinas":
            serie = serie_capacidade_anual(dados_estado).tail(8)
            labels = [str(x) for x in serie.index]
            valores = [round(float(v), 2) for v in serie.values]
            titulo = "MW adicionados"

        elif tipo == "producao_mensal":
            serie = producao_mensal(dados_estado, fonte)
            labels = list(serie.keys())
            valores = [round(float(v), 2) for v in serie.values()]
            titulo = "MWh estimados"

        elif tipo == "producao_anual":
            serie = producao_anual(dados_estado, fonte).tail(8)
            labels = [str(x) for x in serie.index]
            valores = [round(float(v), 2) for v in serie.values]
            titulo = "MWh estimados"

        elif tipo == "projecao_2027":
            serie = serie_capacidade_anual(dados_estado).tail(8)
            previsao = projetar_2027(serie)

            labels = [str(x) for x in serie.index] + ["2027"]
            valores = [round(float(v), 2) for v in serie.values] + [round(previsao, 2)]
            titulo = "MW adicionados/projetados"

        else:
            labels = []
            valores = []
            titulo = "Valores"

        labels_finais = labels

        datasets.append({
            "label": nome_estado(uf),
            "data": valores
        })

    return jsonify({
        "tipo_grafico": "line",
        "labels": labels_finais,
        "datasets": datasets,
        "titulo_eixo": titulo
    })


@app.route("/api/analise")
def api_analise():
    tipo = request.args.get("tipo", "ranking")
    estado = request.args.get("estado", "TODOS")
    fonte = request.args.get("fonte", "Todas")

    df = carregar_dados()
    titulo_fonte = fonte if fonte != "Todas" else "Solar + Eólica"

    if tipo == "ranking":
        dados = filtrar(df, "TODOS", fonte)

        ranking = (
            dados.groupby("sigufprincipal")["potencia_kw"]
            .sum()
            .sort_values(ascending=False) / 1000
        )

        lider_uf = ranking.index[0]
        lider_valor = float(ranking.iloc[0])
        producao_lider = lider_valor * 8760 * fator_capacidade(fonte)

        texto = f"Ranking Nordeste - {titulo_fonte}\n\n"
        texto += f"O estado líder é {nome_estado(lider_uf)}, com {lider_valor:.2f} MW de capacidade instalada.\n"
        texto += f"A produção estimada anual do líder é de aproximadamente {producao_lider:.2f} MWh/ano.\n\n"
        texto += "Classificação dos estados:\n"

        for i, (uf, valor) in enumerate(ranking.items(), start=1):
            texto += f"{i}. {nome_estado(uf)} - {valor:.2f} MW instalados\n"

        return jsonify({"resposta": texto})

    estados = preparar_estados(estado)
    texto = f"Análise selecionada - {titulo_fonte}\n"
    texto += f"Estados comparados: {', '.join(nome_estado(uf) for uf in estados)}\n\n"

    for uf in estados:
        dados = filtrar(df, uf, fonte)
        total = dados["potencia_kw"].sum() / 1000
        prod = total * 8760 * fator_capacidade(fonte)

        texto += f"{nome_estado(uf)}\n"
        texto += f"Capacidade instalada total: {total:.2f} MW\n"
        texto += f"Produção estimada anual: {prod:.2f} MWh/ano\n"
        texto += f"Registros analisados: {len(dados)}\n\n"

    texto += "Observação: a produção foi estimada com base na capacidade instalada e em fator médio de capacidade."

    return jsonify({"resposta": texto})


if __name__ == "__main__":
    print("Servidor iniciado com sucesso.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))