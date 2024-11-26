from langchain.pydantic_v1 import BaseModel, root_validator
from typing import Dict, Any

from sciborg.core.scheduler.base import BaseScheduleTemplate
from sciborg.ai.schema.parameter import ParameterSchemaV1

class ScheduleSchemaV1(BaseModel):
    name: str
    template:  Dict[str, Dict | ParameterSchemaV1]

    @root_validator
    def validate_scheduler(cls, values: Dict[str, Any]) -> Dict:
        BaseScheduleTemplate(**values)
        return values