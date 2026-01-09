from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Literal, Optional

class StudentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    # --- Demographical informations ---
    school: Literal["GP", "MS"] = Field(..., description="Student school")
    sex: Literal["F", "M"] = Field(..., description="Student gender")
    age: int = Field(..., ge=15, le=22)
    address: Literal["U", "R"] = Field(..., description="U: Urban, R: Country")
    famsize: Literal["GT3", "LE3"] = Field(..., description="Taille de la famille")
    Pstatus: Literal["T", "A"] = Field(..., description="Parents living habits")

    # --- Parents works and degree ---
    Medu: int = Field(..., ge=0, le=4, description="Mother degree")
    Fedu: int = Field(..., ge=0, le=4, description="Father degree")
    Mjob: Literal["teacher", "health", "services", "at_home", "other"]
    Fjob: Literal["teacher", "health", "services", "at_home", "other"]
    reason: Literal["home", "reputation", "course", "other"]
    guardian: Literal["mother", "father", "other"]

    # --- Student life habits ---
    traveltime: int = Field(..., ge=1, le=4)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0, le=4)
    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]
    activities: Literal["yes", "no"]
    nursery: Literal["yes", "no"]
    higher: Literal["yes", "no"]
    internet: Literal["yes", "no"]

    # --- Leasure and relationships (Scale 1..5) ---
    famrel: int = Field(..., ge=1, le=5)
    freetime: int = Field(..., ge=1, le=5)
    goout: int = Field(..., ge=1, le=5)
    
    # --- Results and absences ---
    absences: int = Field(..., ge=0, le=93)
    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)

class StepParams(BaseModel):
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