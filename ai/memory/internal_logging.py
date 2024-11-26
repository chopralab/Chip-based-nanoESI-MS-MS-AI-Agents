from langchain.memory.summary import ConversationSummaryMemory, SummarizerMixin
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.language_models import BaseLanguageModel

from typing import Dict, Any, List, Type
import json

from sciborg.ai.prompts.memory import ACTION_SUMMARY_PROMPT, ACTION_LOG_FSA_TEMPLATE

class CustomSummariserMixin(SummarizerMixin):
    """Mixin for summarizer."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = ACTION_SUMMARY_PROMPT
    summary_message_cls: Type[BaseMessage] = SystemMessage

    def predict_new_summary(
        self, messages: List[BaseMessage], existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.predict(summary=existing_summary, new_lines=new_lines)

class CustomActionLogSummaryMemory(ConversationSummaryMemory, CustomSummariserMixin):
    '''
    TODO We need to be able to tune the context window of the summary, it can forget important context

    Custom summary memory for summarization of internal agent thought/action/observation process.
    To use this memory class, the agent executor must return its intermediate steps as an output 
    with the key - `intermediate_steps` (it can be a dictionary or string).

    ### Example
    Human message:
    ```
    'Can you heat the vial for me?'
    ```
    AI actions:
    ```
    - allocate session
    ```
    Summary Memory:
    ```text
    The AI has allocated a session on the instrument with the session ID of f0768a70-e8d0-41c5-ace7-25ad9ad28482. 
    The AI is preparing to heat the vial but needs to ensure that a session has been allocated, a vial has been loaded, the lid is closed, and the heating parameters have been set before proceeding.
    ```
    Human message:
    ```
    'heat vial number 3'
    ```
    AI actions:
    ```
    - open_lid
    - load_vial(3)
    - close_lid
    ```
    Summary Memory:
    ```text
    The AI has received a request to heat vial number 3. 
    The AI has opened the lid, loaded vial number 3 into the microwave synthesizer, and closed the lid in preparation for heating the vial. 
    The current status is that vial number 3 is loaded and the lid is closed.
    ```
    '''
    filtered_tool_list : List[str]
    prompt: BasePromptTemplate = ACTION_SUMMARY_PROMPT
    # prompt: BasePromptTemplate = ACTION_SUMMARY_PROMPT #DELETE
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:

        #filtering
        filtered_tools = [action for action in outputs['intermediate_steps'] if action[0].tool in self.filtered_tool_list]
        copied_outputs = outputs.copy()
        copied_outputs['intermediate_steps'] = str(filtered_tools)
        return super().save_context(inputs, copied_outputs)
    
class FSAMixin(SummarizerMixin):
    '''
    Custom memory mix in for summarization of internal agent thought/action/observation process as
    a finite state automata (FSA). 
    '''
    fsa_object: Type[BaseModel]
    prompt_template: str = ACTION_LOG_FSA_TEMPLATE
    output_parser: JsonOutputParser | None = None
    prompt: BasePromptTemplate | None = None
    
    @root_validator()
    def validate_fsa_mixin(cls, values: Dict):
        # print(values)
        values['output_parser'] = JsonOutputParser(pydantic_object=values['fsa_object'])
        values['prompt'] = PromptTemplate(
            input_variables=['current_fsa', 'internal_operations'],
            partial_variables={'formatting_instructions': values['output_parser'].get_format_instructions()},
            template=values['prompt_template']
        )
        return values

    def predict_new_summary(
        self, 
        messages: List[BaseMessage],
        existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.output_parser
        )
        output = chain.predict(summary=existing_summary, new_lines=new_lines)
        return json.dumps(output, indent=2)

class FSAMemory(ConversationSummaryMemory, FSAMixin):
    '''
    Custom internal memory for summarization of agent's internal thought/action/observation process
    to update psudeo finite state automata representing a microservice controlled by the agent. 
    The psuedo FSA should be passed in as a Pydantic (v1) schema. Use literals to restrict string states
    and ranges to restrict numeric states.

    ### Example
    Schema
    ```python
    class MicrowaveSynthesizerFSA(BaseModel):
        sessionID: str | None = Field(None, description='ID of the session allocation or None if no session allocated')
        lid_status: Literal['open', 'closed'] = Field(default='closed', description='status of the lid')
        vial_status: Literal['loaded', 'unloaded'] = Field(default='status of the vial')
        vial: str | None = Field(default=None, description='Identifier of the vial loaded, None if no vial is loaded')
        heating_status: Literal['not_heating', 'heating'] = Field(default='not_heating', description='status of heating')
        temp: int | None = Field(default=None, description='set tempeature to heat at, None if not currently set')
        duration: int | None = Field(default=None, description='set duration to heat for, None if not currently set')
        pressure: float | None = Field(default=None, description='set pressure to heat at, None if not currently set')
    ```
    FSA state before
    ```json
    {
        "sessionID": "324d4026-a402-4511-b131-c1da515603e1",
        "lid_status": "closed",
        "vial_status": "unloaded",
        "vial": null,
        "heating_status": "not_heating",
        "temp": null,
        "duration": null,
        "pressure": null
    }
    ```
    Human message:
    ```
    'load vial 3'
    ```
    AI actions:
    ```
    - open_lid
    - load_vial(3)
    - close_lid
    ```
    FSA state after
    ```json
    {
        "sessionID": "40d191e2-2f3a-4462-be7b-56e1d87262fd",
        "lid_status": "closed",
        "vial_status": "loaded",
        "vial": "3",
        "heating_status": "not_heating",
        "temp": null,
        "duration": null,
        "pressure": null
    }
    ```
    '''
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        copied_outputs = outputs.copy()
        copied_outputs['intermediate_steps'] = str(copied_outputs['intermediate_steps'])
        return super().save_context(inputs, copied_outputs)