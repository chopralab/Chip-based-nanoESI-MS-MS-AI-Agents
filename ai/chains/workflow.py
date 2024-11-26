'''
Custom LLM chains for workflow construction
'''
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel as BaseModelV1

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLanguageModel

from langchain_openai import ChatOpenAI

from typing import Type

from sciborg.core.library.base import BaseCommandLibrary

from sciborg.ai.schema.command import BaseRunCommandSchemaV1
from sciborg.ai.schema.workflow import RunWorkflowSchemaV1
from sciborg.ai.prompts.workflow import (
    BASE_WORKFLOW_CONSTRUCTION_PROMPT,
    BASE_WORKFLOW_PLANNING_PROMPT,
)

def create_workflow_planner_chain(
    library: BaseCommandLibrary,
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1),
    prompt: str = BASE_WORKFLOW_PLANNING_PROMPT
) -> LLMChain:
    '''
    Creates an LLM chain for workflow planning based on the provided command library.

    This chain can serve as an intermediary between higher level planning and operational planning.
    '''
    # Define prompt template
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=['query'],
        partial_variables={
            'command_library': library.model_dump_json(indent=2)
        }
    )

    # Define LLM chain
    return LLMChain(
        llm=llm,
        prompt=prompt_template,
    )

def create_workflow_constructor_chain(
    library: BaseCommandLibrary,
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1),
    prompt: str = BASE_WORKFLOW_CONSTRUCTION_PROMPT,
    output_schema: Type[BaseModelV1] = RunWorkflowSchemaV1,
) -> LLMChain:
    '''
    Builds a LLM chain for operational workflow construction based on a the provided command library.
    '''
    # Define output parser
    output_parser = JsonOutputParser(pydantic_object=output_schema)

    # Define prompt template
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=['query'],
        partial_variables={
            "command_library": library.model_dump_json(indent=2),
            "format_instructions": output_parser.get_format_instructions(),
            }
    )

    # Define LLM chain
    return LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_parser=output_parser,
    )