"""
Agente de IA Generativa — "tradutor" da predição em linguagem natural.

Usa a API REST do Gemini (Google AI Studio) para transformar a saída bruta
do modelo de Machine Learning em uma explicação clara, fundamentada nos
dados do paciente e nas métricas reais do modelo — SEM alucinar diagnósticos,
recomendações médicas ou números que não foram fornecidos.

Como obter a chave da API (gratuita):
1. Acesse https://aistudio.google.com/apikey
2. Faça login com uma conta Google
3. Clique em "Create API key"
4. Copie a chave e defina a variável de ambiente GEMINI_API_KEY
   (ou coloque em um arquivo .env na pasta backend/, veja .env.example)
"""

import os
import requests

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = """Você é um assistente educacional que explica, em linguagem simples e
acolhedora, o resultado de um modelo de Machine Learning (Naive Bayes) treinado para
estimar a probabilidade de doença cardíaca a partir de exames clínicos.

REGRAS OBRIGATÓRIAS:
1. Você NÃO é um médico e NÃO está fazendo um diagnóstico. Sempre deixe claro que o
   resultado é uma estimativa estatística de um modelo de IA, não um diagnóstico médico,
   e que a pessoa deve procurar um cardiologista para avaliação real.
2. Use APENAS os dados fornecidos (predição, probabilidades, variáveis do paciente e
   métricas do modelo). NUNCA invente exames, históricos, causas ou recomendações de
   tratamento/medicação que não foram informados.
3. Explique de forma simples quais variáveis de entrada mais chamam atenção no caso
   (ex: colesterol alto, idade, pressão), mas apenas com base nos valores recebidos —
   sem afirmar relações de causa e efeito que o modelo não comprova.
4. Mencione, de forma breve, o nível de confiabilidade do modelo (acurácia,
   sensibilidade e especificidade fornecidas), para que o usuário entenda que o
   resultado não é 100% certo.
5. Tom: claro, respeitoso, sem termos técnicos excessivos, em português do Brasil.
6. Tamanho: no máximo 2 parágrafos curtos.
7. Nunca alucine números, estudos ou fatos que não foram passados a você no prompt.
"""


class GeminiAgentError(Exception):
    """Erro ao chamar a API do Gemini."""


def montar_prompt_usuario(paciente: dict, resultado: dict, metricas: dict) -> str:
    """Monta o prompt com todos os dados reais (paciente + predição + métricas)."""
    return f"""Dados do paciente informados:
{_formatar_dict(paciente)}

Resultado do modelo de Machine Learning (Naive Bayes):
- Predição: {resultado['rotulo']}
- Probabilidade de NÃO ter doença cardíaca: {resultado['probabilidade_sem_doenca']*100:.1f}%
- Probabilidade de TER doença cardíaca: {resultado['probabilidade_com_doenca']*100:.1f}%

Métricas de desempenho do modelo (avaliadas no conjunto de teste):
- Acurácia: {metricas['acuracia']*100:.1f}%
- Sensibilidade (capacidade de detectar quem tem a doença): {metricas['sensibilidade']*100:.1f}%
- Especificidade (capacidade de identificar quem não tem a doença): {metricas['especificidade']*100:.1f}%
- Precisão: {metricas['precisao']*100:.1f}%

Explique esse resultado para o usuário seguindo todas as regras do system prompt."""


def _formatar_dict(d: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


def gerar_explicacao(paciente: dict, resultado: dict, metricas: dict) -> str:
    """Chama a API do Gemini e retorna a explicação em texto. Lança GeminiAgentError em falha."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiAgentError(
            "GEMINI_API_KEY não configurada. Defina a variável de ambiente "
            "ou crie um arquivo .env na pasta backend/ (veja .env.example)."
        )

    prompt_usuario = montar_prompt_usuario(paciente, resultado, metricas)

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt_usuario}]}],
        "generationConfig": {
            "temperature": 0.3,  # baixa temperatura: respostas mais factuais, menos "criativas"
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingBudget": 0},  # desliga o "raciocínio" interno, que consumia os tokens de saída
        },
    }

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise GeminiAgentError(f"Falha ao chamar a API do Gemini: {e}") from e

    data = resp.json()
    try:
        candidato = data["candidates"][0]
        texto = candidato["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise GeminiAgentError(f"Resposta inesperada da API do Gemini: {data}") from e

    if candidato.get("finishReason") == "MAX_TOKENS" and not texto:
        raise GeminiAgentError(
            "A resposta foi cortada antes de gerar texto (limite de tokens atingido). "
            "Tente novamente."
        )

    return texto