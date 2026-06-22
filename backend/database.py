"""
Banco de dados SQLite para persistência do histórico de predições.

Salva por consulta:
- Dados de entrada do paciente (todas as 13 variáveis)
- Resultado da predição (rótulo + probabilidades)
- Explicação gerada pelo agente Gemini
- Timestamp da consulta

O arquivo historico.db é criado automaticamente na pasta backend/
na primeira vez que o backend é iniciado.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "historico.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predicoes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    -- dados do paciente
    age         INTEGER, sex      INTEGER, cp       INTEGER,
    trestbps    INTEGER, chol     INTEGER, fbs      INTEGER,
    restecg     INTEGER, thalach  INTEGER, exang    INTEGER,
    oldpeak     REAL,    slope    INTEGER, ca       INTEGER,
    thal        INTEGER,
    -- resultado
    predicao                  INTEGER NOT NULL,
    rotulo                    TEXT    NOT NULL,
    probabilidade_sem_doenca  REAL    NOT NULL,
    probabilidade_com_doenca  REAL    NOT NULL,
    modelo_utilizado          TEXT    NOT NULL,
    explicacao_ia             TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # permite acessar colunas por nome
    return conn


def inicializar_banco() -> None:
    """Cria a tabela se ainda não existir. Chamado na inicialização do backend."""
    with _connect() as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()


def salvar_predicao(dados_paciente: dict, resultado: dict, explicacao: str) -> int:
    """
    Insere uma nova linha no banco com os dados do paciente + resultado + explicação.
    Retorna o id gerado.
    """
    sql = """
    INSERT INTO predicoes (
        timestamp,
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal,
        predicao, rotulo, probabilidade_sem_doenca, probabilidade_com_doenca,
        modelo_utilizado, explicacao_ia
    ) VALUES (
        :timestamp,
        :age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, :thalach, :exang,
        :oldpeak, :slope, :ca, :thal,
        :predicao, :rotulo, :probabilidade_sem_doenca, :probabilidade_com_doenca,
        :modelo_utilizado, :explicacao_ia
    )
    """
    params = {
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        **dados_paciente,
        **resultado,
        "explicacao_ia": explicacao,
    }
    with _connect() as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.lastrowid


def buscar_historico(limite: int = 20) -> list[dict]:
    """
    Retorna as últimas `limite` predições em ordem decrescente de data.
    """
    sql = """
    SELECT
        id, timestamp, age, sex, predicao, rotulo,
        probabilidade_com_doenca, explicacao_ia
    FROM predicoes
    ORDER BY id DESC
    LIMIT ?
    """
    with _connect() as conn:
        rows = conn.execute(sql, (limite,)).fetchall()
    return [dict(row) for row in rows]


def buscar_predicao_por_id(pred_id: int) -> dict | None:
    """Retorna todos os campos de uma predição específica pelo id."""
    sql = "SELECT * FROM predicoes WHERE id = ?"
    with _connect() as conn:
        row = conn.execute(sql, (pred_id,)).fetchone()
    return dict(row) if row else None