# from langchain.pydantic_v1 import BaseModel, root_validator
from pydantic import BaseModel as BaseModelV2, model_validator
from typing import Dict, Any

from sciborg_dev.core.scheduler.base import BaseScheduleTemplate
from sciborg_dev.ai.schema.parameter import ParameterSchemaV1

class ScheduleSchemaV1(BaseModelV2):
    name: str
    template:  Dict[str, Dict | ParameterSchemaV1]

    @model_validator(mode='before')
    def validate_scheduler(cls, values: Dict[str, Any]) -> Dict:
        BaseScheduleTemplate(**values)
        return values