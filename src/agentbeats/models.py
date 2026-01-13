from typing import Any
from pydantic import BaseModel, HttpUrl

class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl] # role-endpoint mapping
    config: dict[str, Any]

class EvalResult(BaseModel):
    total_instances: int
    total_resolved: int
    resolution_rate: float
    fail_to_pass_failed_rate: float
    pass_to_pass_passed_rate: float
