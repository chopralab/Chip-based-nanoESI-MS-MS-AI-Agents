# from langchain.pydantic_v1 import BaseModel, root_validator
from pydantic import BaseModel as BaseModelV2
from typing import List, Any, Dict

from sciborg_dev.core.workflow.base import BaseInfoWorkflow
from sciborg_dev.ai.schema.command import BaseRunCommandSchemaV1

class RunWorkflowSchemaV1(BaseModelV2):
    name: str
    commands: List[BaseRunCommandSchemaV1]