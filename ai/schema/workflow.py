from langchain.pydantic_v1 import BaseModel, root_validator
from typing import List, Any, Dict

from sciborg.core.workflow.base import BaseInfoWorkflow
from sciborg.ai.schema.command import BaseRunCommandSchemaV1

class RunWorkflowSchemaV1(BaseModel):
    name: str
    commands: List[BaseRunCommandSchemaV1]