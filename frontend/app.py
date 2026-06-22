"""
Frontend — Agente Preditivo Especialista (Heart Disease)

Interface em Streamlit onde o usuário insere os dados clínicos de um paciente e visualiza:
    1. O resultado bruto do modelo (predição + probabilidades)
    2. A explicação em linguagem natural gerada pelo agente de IA (Gemini)
    3. O histórico das últimas consultas salvas no banco de dados

Para rodar (com o backend já no ar em outra aba do terminal):
    streamlit run app.py
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Agente Preditivo — Doença Cardíaca",
    page_icon="🫀",
    layout="centered",
)

st.title("🫀 Agente Preditivo Especialista")
st.caption("Predição de risco de doença cardíaca com Machine Learning + explicação por IA generativa")

# ---------------------------------------------------------------------------
# Sidebar: info do modelo + histórico de consultas
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Sobre o modelo")
    try:
        info = requests.get(f"{API_URL}/info", timeout=5).json()
        st.success("Backend conectado ✅")
        st.markdown(f"**Modelo em produção:** {info['modelo_utilizado']}")
        m = info["metricas_teste"]
        st.markdown(
            f"""
- **Acurácia:** {m['acuracia']*100:.1f}%
- **Sensibilidade:** {m['sensibilidade']*100:.1f}%
- **Especificidade:** {m['especificidade']*100:.1f}%
- **Precisão:** {m['precisao']*100:.1f}%
"""
        )
        with st.expander("Comparar todos os algoritmos testados"):
            st.dataframe(info["comparacao_todos_os_modelos"], use_container_width=True)
    except requests.exceptions.RequestException:
        st.error("⚠️ Backend não está respondendo. Rode `uvicorn main:app --reload` na pasta backend/.")
        info = None

    st.divider()

    # Histórico de consultas
    st.header("🕑 Histórico de consultas")
    try:
        historico = requests.get(f"{API_URL}/historico?limite=10", timeout=5).json()
        if not historico:
            st.caption("Nenhuma consulta realizada ainda.")
        else:
            for reg in historico:
                icone = "⚠️" if reg["predicao"] == 1 else "✅"
                prob = reg["probabilidade_com_doenca"] * 100
                with st.expander(f"{icone} #{reg['id']} — {reg['timestamp']} — {prob:.0f}% risco"):
                    st.markdown(f"**Idade:** {reg['age']} anos")
                    st.markdown(f"**Resultado:** {reg['rotulo']}")
                    st.markdown(f"**Explicação da IA:**")
                    st.caption(reg["explicacao_ia"])
    except requests.exceptions.RequestException:
        st.caption("Histórico indisponível.")

    st.divider()
    st.caption(
        "⚠️ Esta ferramenta é educacional e **não substitui** uma avaliação médica. "
        "Os resultados são estimativas estatísticas de um modelo de IA."
    )

# ---------------------------------------------------------------------------
# Formulário de entrada
# ---------------------------------------------------------------------------
st.subheader("Dados do paciente")

with st.form("formulario_paciente"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Idade", min_value=1, max_value=120, value=58)
        sex = st.selectbox("Sexo", options=[("Masculino", 1), ("Feminino", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Tipo de dor no peito",
            options=[
                ("Típica (0)", 0), ("Atípica (1)", 1),
                ("Não-anginosa (2)", 2), ("Assintomática (3)", 3),
            ],
            format_func=lambda x: x[0],
        )[1]
        trestbps = st.number_input("Pressão arterial em repouso (mmHg)", min_value=60, max_value=250, value=140)
        chol = st.number_input("Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=289)
        fbs = st.selectbox("Glicemia de jejum > 120 mg/dl?", options=[("Não", 0), ("Sim", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox(
            "Resultado do ECG em repouso",
            options=[("Normal (0)", 0), ("Anormalidade ST-T (1)", 1), ("Hipertrofia ventricular (2)", 2)],
            format_func=lambda x: x[0],
        )[1]

    with col2:
        thalach = st.number_input("Frequência cardíaca máxima (bpm)", min_value=60, max_value=250, value=145)
        exang = st.selectbox("Angina induzida por exercício?", options=[("Não", 0), ("Sim", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("Depressão do segmento ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.8, step=0.1)
        slope = st.selectbox(
            "Inclinação do segmento ST",
            options=[("Ascendente (0)", 0), ("Plana (1)", 1), ("Descendente (2)", 2)],
            format_func=lambda x: x[0],
        )[1]
        ca = st.selectbox("Nº de vasos principais (fluoroscopia)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Talassemia",
            options=[("Normal (0)", 0), ("Defeito fixo (1)", 1), ("Defeito reversível (2)", 2), ("Outro (3)", 3)],
            format_func=lambda x: x[0],
        )[1]

    enviado = st.form_submit_button("🔍 Analisar", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Chamada à API e exibição dos resultados
# ---------------------------------------------------------------------------
if enviado:
    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    with st.spinner("Consultando o modelo e gerando explicação..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=40)
            resp.raise_for_status()
            resultado = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao consultar a API: {e}")
            st.stop()

    st.divider()
    st.subheader("📊 Resultado do modelo")

    col_a, col_b = st.columns(2)
    if resultado["predicao"] == 1:
        col_a.metric("Predição", "⚠️ Risco detectado")
    else:
        col_a.metric("Predição", "✅ Sem indícios")
    col_b.metric("Modelo utilizado", resultado["modelo_utilizado"])

    st.progress(resultado["probabilidade_com_doenca"], text=f"Probabilidade de doença cardíaca: {resultado['probabilidade_com_doenca']*100:.1f}%")
    st.progress(resultado["probabilidade_sem_doenca"], text=f"Probabilidade de ausência de doença: {resultado['probabilidade_sem_doenca']*100:.1f}%")

    st.subheader("🤖 Explicação do agente de IA")
    st.info(resultado["explicacao_ia"])

    st.caption(
        "Lembrete: este resultado é uma estimativa estatística baseada em um modelo "
        "treinado com dados históricos. Não substitui consulta, exame ou diagnóstico médico."
    )