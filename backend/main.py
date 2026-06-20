"""
Backend FastAPI — Agente Preditivo Especialista (Heart Disease).

Endpoints:
    GET  /                -> health check
    GET  /info             -> metadados do modelo (algoritmo escolhido, métricas)
    POST /predict           -> recebe dados do paciente, retorna predição + explicação da IA

Para rodar localmente:
    uvicorn main:app --reload --port 8000

Documentação interativa automática em:
    http://localhost:8000/docs
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PacienteInput, PredicaoOutput
from gemini_agent import gerar_explicacao, GeminiAgentError

load_dotenv()  # carrega GEMINI_API_KEY do arquivo .env, se existir

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

app = FastAPI(
    title="Agente Preditivo Especialista — Heart Disease",
    description=(
        "API que recebe exames de um paciente, prediz a probabilidade de doença "
        "cardíaca usando um modelo de Machine Learning, e explica o resultado em "
        "linguagem natural usando um agente de IA generativa (Gemini)."
    ),
    version="1.0.0",
)

# Libera acesso do frontend (Streamlit roda em outra porta/processo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Carregamento do modelo, scaler e metadados (uma única vez, na subida da API)
# ---------------------------------------------------------------------------
try:
    modelo = joblib.load(ARTIFACTS_DIR / "modelo_final.pkl")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    with open(ARTIFACTS_DIR / "metadados.json", encoding="utf-8") as f:
        metadados = json.load(f)
except FileNotFoundError as e:
    raise RuntimeError(
        "Artefatos do modelo não encontrados em backend/artifacts/. "
        "Execute o notebook notebooks/Vanessa_e_Ana_Ajuste_Heart_Disease_IA.ipynb "
        "até o final para gerar modelo_final.pkl, scaler.pkl e metadados.json."
    ) from e

COLUNAS_ENTRADA = metadados["colunas_entrada"]
METRICAS_MODELO = metadados["metricas_teste"]
NOME_MODELO = metadados["modelo_escolhido"]


@app.get("/")
def root():
    return {"status": "ok", "mensagem": "API do Agente Preditivo Especialista está no ar."}


@app.get("/info")
def info():
    """Retorna qual modelo foi escolhido e suas métricas de avaliação."""
    return {
        "modelo_utilizado": NOME_MODELO,
        "metricas_teste": METRICAS_MODELO,
        "comparacao_todos_os_modelos": metadados["todas_as_metricas"],
        "colunas_esperadas": COLUNAS_ENTRADA,
    }


@app.post("/predict", response_model=PredicaoOutput)
def predict(paciente: PacienteInput):
    """
    Recebe os dados de um paciente, roda o modelo treinado e pede ao agente
    de IA (Gemini) para explicar o resultado em linguagem natural.
    """
    dados = paciente.model_dump()

    # Garante a mesma ordem de colunas usada no treino
    entrada = np.array([[dados[col] for col in COLUNAS_ENTRADA]])
    entrada_normalizada = scaler.transform(entrada)

    predicao = int(modelo.predict(entrada_normalizada)[0])
    probabilidades = modelo.predict_proba(entrada_normalizada)[0]

    resultado = {
        "predicao": predicao,
        "rotulo": "Risco de doença cardíaca" if predicao == 1 else "Sem indícios de doença cardíaca",
        "probabilidade_sem_doenca": float(probabilidades[0]),
        "probabilidade_com_doenca": float(probabilidades[1]),
        "modelo_utilizado": NOME_MODELO,
    }

    try:
        explicacao = gerar_explicacao(dados, resultado, METRICAS_MODELO)
    except GeminiAgentError as e:
        # A predição (núcleo do produto) continua funcionando mesmo se a IA falhar
        explicacao = (
            "[Explicação por IA indisponível no momento] "
            f"Motivo técnico: {e}. "
            "O resultado bruto do modelo continua válido e está descrito acima."
        )

    return PredicaoOutput(**resultado, explicacao_ia=explicacao)
