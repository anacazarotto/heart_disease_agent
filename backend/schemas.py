"""
Schemas Pydantic para validação dos dados de entrada/saída da API.

As variáveis e seus intervalos seguem o dicionário de dados do dataset
Heart Disease (Kaggle/UCI), documentado também no notebook de exploração.
"""

from pydantic import BaseModel, Field


class PacienteInput(BaseModel):
    """Dados de um paciente para predição de doença cardíaca."""

    age: int = Field(..., ge=1, le=120, description="Idade do paciente (anos)")
    sex: int = Field(..., ge=0, le=1, description="Sexo: 1 = masculino, 0 = feminino")
    cp: int = Field(..., ge=0, le=3, description="Tipo de dor no peito (0-3)")
    trestbps: int = Field(..., ge=60, le=250, description="Pressão arterial em repouso (mmHg)")
    chol: int = Field(..., ge=100, le=600, description="Colesterol sérico (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Glicemia de jejum > 120 mg/dl: 1 = sim, 0 = não")
    restecg: int = Field(..., ge=0, le=2, description="Resultado do ECG em repouso (0-2)")
    thalach: int = Field(..., ge=60, le=250, description="Frequência cardíaca máxima atingida (bpm)")
    exang: int = Field(..., ge=0, le=1, description="Angina induzida por exercício: 1 = sim, 0 = não")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="Depressão do segmento ST")
    slope: int = Field(..., ge=0, le=2, description="Inclinação do segmento ST no pico do exercício (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Número de vasos principais coloridos por fluoroscopia (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Talassemia (0=normal, 1=defeito fixo, 2=defeito reversível)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 58, "sex": 1, "cp": 0, "trestbps": 140, "chol": 289,
                "fbs": 0, "restecg": 0, "thalach": 145, "exang": 1,
                "oldpeak": 1.8, "slope": 1, "ca": 1, "thal": 2
            }
        }


class PredicaoOutput(BaseModel):
    """Resultado bruto do modelo + explicação gerada pelo agente de IA."""

    predicao: int
    rotulo: str
    probabilidade_sem_doenca: float
    probabilidade_com_doenca: float
    modelo_utilizado: str
    explicacao_ia: str
