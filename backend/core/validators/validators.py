from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class StudentInput(BaseModel):
    school: str = Field(..., example="GP")
    sex: str = Field(..., example="F")
    age: int = Field(..., ge=15, le=22)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0)
    absences: int = Field(..., ge=0)
    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)

class StepParams(BaseModel):
    # Flexible pour accepter les params de DataLoader, Outliers, etc.
    files_to_load: Optional[Dict[str, str]] = None
    sep: Optional[str] = ";"
    contamination: Optional[float] = 0.01
    target_columns: Optional[List[str]] = None
    max_iter: Optional[int] = 10

class PipelineStep(BaseModel):
    step: str
    strategy: Optional[str] = None
    params: Optional[StepParams] = None

class ScenarioConfig(BaseModel):
    label: str
    exclusions: List[str]

class LearningConfig(BaseModel):
    strategies: List[str]
    scenarii: List[ScenarioConfig]

class FullPipelineConfig(BaseModel):
    pipeline: Dict[str, List[PipelineStep]]
    learning: LearningConfig

class PredictionResponse(BaseModel):
    prediction_id: str
    model_type: str
    is_failure: bool
    probability: float
    timestamp: datetime