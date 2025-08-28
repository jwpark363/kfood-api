# prediction.py: Pydantic 모델 정의(입출력 스키마)

from pydantic import BaseModel

class PredictionResponse(BaseModel):
    class_idx: int
    class_name: str
    probability: float