from typing import Any
from pydantic import BaseModel, HttpUrl

class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl] # role-endpoint mapping
    config: dict[str, Any]

class EvalResult(BaseModel):
    total_instances: int
    resolved_pct: float
    breaking_resolved_pct: float
    partially_resolved_pct: float
    work_in_progress_pct: float
    regression_pct: float
    no_op_pct: float
    error_pct: float
    fail_to_pass_passed_pct: float
    pass_to_pass_passed_pct: float
    green_agent_model: str
    
