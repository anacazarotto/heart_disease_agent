# 🫀 Agente Preditivo Especialista — Heart Disease

Projeto que une **Machine Learning** e **Inteligência Artificial Generativa** para estimar o
risco de doença cardíaca a partir de exames clínicos, e explicar o resultado em linguagem
natural através de um agente de IA (Gemini).

> **Integrantes:** Ana Carla Londero Cazarotto e Vanessa da Silva
> **Dataset:** [Heart Disease (Kaggle/UCI)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## 📐 Arquitetura do projeto

```
heart-disease-agent/
├── data/
│   └── heart.csv                  # dataset original
├── notebooks/
│   └── Vanessa_e_Ana_Ajuste_Heart_Disease_IA.ipynb   # Etapa A: EDA + treino + comparação
├── backend/
│   ├── main.py                    # API FastAPI (endpoints /, /info, /predict)
│   ├── schemas.py                 # validação dos dados de entrada (Pydantic)
│   ├── gemini_agent.py            # integração com a API do Gemini
│   ├── artifacts/                 # modelo_final.pkl, scaler.pkl, metadados.json
│   ├── requirements.txt
│   └── .env.example               # modelo do arquivo de variáveis de ambiente
├── frontend/
│   ├── app.py                     # interface Streamlit
│   └── requirements.txt
├── reports/
│   └── figures/                   # gráficos exportados em PDF (Seaborn + matrizes de confusão)
└── README.md
```

### Como as peças se conectam

```
[Notebook] --treina e exporta--> [backend/artifacts/*.pkl, *.json]
                                          │
                                          ▼
[Frontend Streamlit] --POST /predict--> [Backend FastAPI] --chama--> [API Gemini]
        (usuário insere dados)          (roda o modelo)        (explica o resultado)
```

1. O **notebook** (Etapa A) faz a exploração de dados, treina 4 algoritmos
   (Regressão Logística, KNN, MLP e Naive Bayes), compara as métricas e exporta
   automaticamente o **melhor modelo** para `backend/artifacts/`.
2. O **backend FastAPI** (Etapa B) carrega esse modelo, expõe o endpoint `POST /predict`,
   e chama a **API do Gemini** com um *system prompt* que instrui o agente a explicar o
   resultado de forma clara e **sem alucinar** diagnósticos ou dados não fornecidos.
3. O **frontend Streamlit** (Etapa C) é a interface onde o usuário insere os dados do
   paciente e visualiza tanto o resultado bruto do modelo quanto a explicação da IA.

---

## 🚀 Como rodar o projeto

### Pré-requisitos
- Python 3.10+
- Uma chave de API gratuita do Gemini (veja o passo a passo abaixo)

### 1. Clonar o repositório
```bash
git clone <url-do-repositorio>
cd heart-disease-agent
```

### 2. Gerar uma chave da API do Gemini (gratuita)
1. Acesse [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Faça login com uma conta Google
3. Clique em **"Create API key"**
4. Copie a chave gerada

### 3. Configurar o backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Abra o arquivo .env e cole sua chave:
# GEMINI_API_KEY=sua_chave_aqui
```

> ⚠️ Os arquivos `modelo_final.pkl`, `scaler.pkl` e `metadados.json` já vêm prontos em
> `backend/artifacts/`, gerados pelo notebook. Se quiser retreinar do zero, basta rodar
> o notebook `notebooks/Vanessa_e_Ana_Ajuste_Heart_Disease_IA.ipynb` até o final — ele
> sobrescreve esses arquivos automaticamente.

### 4. Subir o backend
```bash
# ainda dentro da pasta backend/, com o venv ativado
uvicorn main:app --reload --port 8000
```
- API disponível em: `http://localhost:8000`
- Documentação interativa (Swagger): `http://localhost:8000/docs`

### 5. Subir o frontend (em outro terminal)
```bash
cd frontend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```
- Interface disponível em: `http://localhost:8501`

### 6. Usar
Com o backend e o frontend rodando, abra `http://localhost:8501` no navegador, preencha
os dados do paciente e clique em **"Analisar"**.

---

## 🧠 Sobre o modelo

Quatro algoritmos de classificação foram treinados e comparados (acurácia, sensibilidade,
especificidade, precisão e F1-score). O melhor modelo é selecionado **automaticamente**
pelo notebook (maior acurácia no conjunto de teste) e exportado para uso no backend.

| Algoritmo | Observação |
|---|---|
| Regressão Logística | Baseline linear interpretável |
| KNN | Melhor K escolhido via GridSearchCV (cross-validation) |
| MLP (Rede Neural) | Melhor arquitetura/ativação escolhida via GridSearchCV |
| **Naive Bayes** | Classificador probabilístico gaussiano *(novo nesta etapa)* |

As métricas detalhadas de cada algoritmo, os gráficos exploratórios (Seaborn) e a análise
crítica completa estão no notebook e no Relatório Técnico (PDF).

---

## 🤖 Sobre o agente de IA

O agente usa a API do **Gemini** (`gemini-2.0-flash`) com um *system prompt* que:
- Deixa claro que o resultado é uma **estimativa estatística**, não um diagnóstico médico;
- Usa **apenas** os dados realmente fornecidos (predição, probabilidades, valores do
  paciente e métricas reais do modelo) — instruído a **nunca inventar** exames, causas
  ou recomendações de tratamento;
- Roda com temperatura baixa (0.3) para priorizar respostas factuais;
- Possui fallback: se a API do Gemini falhar ou a chave não estiver configurada, o
  backend ainda retorna o resultado bruto do modelo normalmente.

---

## 📓 Diário de Bordo de Contribuições
 
### Vanessa
- Revisei a base e os gráficos do Seaborn do trabalho anterior, e implementei o Naive Bayes.
- Rodei o GridSearch no KNN e no MLP de novo pra comparar tudo nas mesmas condições.
- Montei a tabela comparativa das métricas e escrevi a análise crítica dos quatro algoritmos.
- Ajustes finais no notebook e exportação do modelo (Naive Bayes foi o que saiu na frente).
### Ana
- Criei a chave do Gemini e montei o backend em FastAPI com a validação dos dados de entrada.
- Integrei o modelo exportado com o endpoint de predição e comecei a escrever o prompt do agente.
- Resolvi uns perrengues com a API do Gemini (modelo tinha sido descontinuado, resposta vinha cortada) e fechei o frontend em Streamlit.
- Testes ponta a ponta, README e organização do repositório no GitHub.
---

## ⚠️ Aviso

Este projeto é **acadêmico e educacional**. Os resultados gerados pelo modelo e pelo
agente de IA **não substituem** avaliação, diagnóstico ou tratamento médico profissional.
