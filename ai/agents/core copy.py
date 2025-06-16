from typing import Any, Callable, Dict, Type, Literal
from inspect import signature, getdoc
from pydantic import BaseModel as BaseModelV2

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.memory import BaseMemory

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.prompts import ChatPromptTemplate, PromptTemplate, BasePromptTemplate
from langchain.tools import StructuredTool
from langchain.tools.human import HumanInputRun
# from langchain.pydantic_v1 import BaseModel as BaseModelV1
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory

from langchain_openai import ChatOpenAI
from langchain.agents import tool

from sciborg_dev.core.library.base import BaseDriverMicroservice
from sciborg_dev.core.command.base import BaseDriverCommand
from sciborg_dev.ai.memory.internal_logging import CustomActionLogSummaryMemory, FSAMemory
from sciborg_dev.ai.memory.embedding import EmbeddingSummaryMemory
from sciborg_dev.ai.prompts.agent import (
    HUMAN_TOOL_INSTRUCTIONS,
    ASSUME_DEFAULTS_INSTRUCTIONS,
    BASE_LINQX_CHAT_PROMPT_TEMPLATE,
    RAG_AS_A_TOOL_INSTRUCTIONS
)
from sciborg_dev.ai.tools.core import LinqxTool
from sciborg_dev.ai.agents.rag_agent import rag_agent
# from sciborg_dev.ai.agents.pubchem_agent import pubchem_agent #we build this from the drive file

def command_to_tool(
    command: BaseDriverCommand,
    schema: type[BaseModelV2] = None
) -> StructuredTool:
    '''
    Helper function to convert BaseDriverCommand to a Langchain tool
    
    Note, this disables LINQX validation but still allows for internal validation
    '''
    return StructuredTool.from_function(
        name=command.name,
        args_schema=schema,
        description=f"Function Signature:\n{signature(command._function)}\nFunction Docstring:\n{getdoc(command._function)}",
        func=command._function,
        handle_tool_error=True,
    )

def create_linqx_chat_agent(
        microservice: BaseDriverMicroservice,
        llm: BaseLanguageModel = ChatOpenAI(temperature=0),
        prompt_template: str = BASE_LINQX_CHAT_PROMPT_TEMPLATE,
        use_memory: Literal['chat', 'action', 'embedding', 'all'] | None = None,
        memory: BaseMemory | None = None,
        intermediate_memory_buffer: str = "",
        past_action_log: str = "",
        human_interaction: bool = True,
        assume_defaults: bool = False,
        rag_vectordb_path: str = None,
        agent_description: str | None = None,
        agent_as_a_tool: AgentExecutor | None = None,
        agent_as_a_fsa: bool = False,
        fsa_object: Type[BaseModelV2] | None = None,
        start_state: BaseModelV2 | None = None,
        use_linqx_tools: bool = True,
        handle_tool_error: bool = True,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        **agent_executor_kwargs
) -> AgentExecutor:
    '''
    Builds an AI agent (Langchain `AgentExecutor` object) to operate the provided microservice based on user requests

    Args\n
    - microservice - The microservice (driver side) that the agent will operate
    - llm - The LLM to build the agent with, defaults to gpt-3.5-turbo
    - prompt_template - The prompt template to provide the agent with
        - Prompt template must include the following placeholders for formatting\n
        `['tools', 'tool_names', 'agent_scratchpad', 'chat_history', 'microservice', 'microservice description']`
    - use_memory - Boolean flag for using memory with an agent
    - memory - The memory object of the agent, defaults to conversation buffer memory with a window of 5
    - past_action_log - A log of past actions and observations that the agent has performed
    - human_interaction - Allows for the agent to interact with a human during operation, does not assume deafaults and requests human for additional information
    - assume_defaults - Assumes the agent is being used as a 'tool' by other agents, does not assume defaults and requests additional information in response
    - verbose - The verbosity of the agent
    - **agent_executor_kwargs - Any additional kwargs to pass to the AgentExecutor upon construction

    Returns\n
    - AgentExecutor - An AI agent which operates the provided microservice upon request

    Example\n
    ```python
    # Build driver side microservice
    my_microservice = module_to_microservice(my_module)
    # Build agent executor
    linqx_agent = create_linqx_chat_agent(microservice = my_microservice)
    # query the agent
    linqx_agent.invoke({'input': 'Please help me with ...'})
    ```
    '''

    # We the agent cannot interact with a human and be used as a tool at the same time
    if human_interaction and assume_defaults:
        raise ValueError(
            """Agent is not desiged to communicate with a human and be used as a tool at the same time.
            If you intend to use the agent as a tool and need human interaction, have a managing agent set up for human interaction use this agent as a tool."""
        )

    # Define input and partial variables
    input_variables = ['tools', 'tool_names', 'agent_scratchpad']
    partial_variables = {
        'microservice': microservice.name,
        'microservice_description': microservice.desc,
    }


    #TODO: do we assign the loaded buffers here? 
    # Add memory if needed to prompt
    if use_memory == 'all':
        # If we are using memory, this will be provide by agent executor
        input_variables.append('chat_history') 
        input_variables.append('past_action_log')
        input_variables.append('embedding_log')
    elif use_memory == 'chat':
        input_variables.append('chat_history')
        partial_variables['past_action_log'] = past_action_log
    elif use_memory == 'action':
         input_variables.append('past_action_log')
         partial_variables['chat_history'] = ""
    elif use_memory == 'embedding' and rag_vectordb_path:
        input_variables.append('embedding_log')
    else:
        # Else use can provide a custom log if needed
        partial_variables['chat_history'] = ""
        partial_variables['past_action_log'] = past_action_log

    # Adding instructions
    TOTAL_INSTRUCTIONS = ""
    if human_interaction:
        TOTAL_INSTRUCTIONS += HUMAN_TOOL_INSTRUCTIONS
    if assume_defaults:
        TOTAL_INSTRUCTIONS += ASSUME_DEFAULTS_INSTRUCTIONS
    if rag_vectordb_path:
        TOTAL_INSTRUCTIONS += RAG_AS_A_TOOL_INSTRUCTIONS
    
    partial_variables['additional_instructions'] = TOTAL_INSTRUCTIONS

    # Build prompt from template
    prompt = PromptTemplate(
        input_variables=input_variables,
        partial_variables=partial_variables,
            template=prompt_template,
        )

    # Build tools from microservice commands
    if use_linqx_tools:
        tools = [LinqxTool(linqx_command=command, handle_tool_error=handle_tool_error) for command in microservice.commands.values()]
    else:
        tools = [command_to_tool(command) for command in microservice.commands.values()]

    action_tool_names = [x.name for x in tools]

    if human_interaction: 
        tools.append(HumanInputRun())
    
    if rag_vectordb_path: 
        @tool
        def call_RAG_agent(question: str) -> str:
            """
            This is the function that will query the relevant sources of information to get the answer to the question.
            There will be situations when you are not able to answer the question directly from the information you currently have. In such cases, you can search for the answer in the relevant sources of information.
            Often the user will also specify that you need to refer to "information" or "documents" to get the answer.
            TAKSK: You have to frame the best possible "question" that is extremely descriptive and then use it as a parameter to query the relevant sources of information and return the citations if present.
            """
            RAG_agent = rag_agent(question, rag_vectordb_path)
            store_output = RAG_agent.invoke({"question": question})
            return store_output['output']
        
        tools.append(call_RAG_agent)

    #TODO: we do not want to continue doing this. Need to have a better way to build multi agent architectures
    if agent_as_a_tool is not None:

        @tool
        def call_provided_Agent(question: str) -> str:
            "This is the function that will call the provided agent as a tool."
            output = agent_as_a_tool.invoke({"input": question})
            return output['output']
        tools.append(call_provided_Agent)

    # Assign agent memory

    context_tool_names = list(set([x.name for x in tools]) - set(action_tool_names))

    if use_memory == 'both' and memory is None and not agent_as_a_fsa:
        chat_memory = ConversationBufferWindowMemory(memory_key='chat_history', input_key='input', output_key='output')
        intermediate_memory =  CustomActionLogSummaryMemory(llm=ChatOpenAI(temperature=0, model='gpt-4'), memory_key='past_action_log', input_key='input', output_key='intermediate_steps', filtered_tool_list=action_tool_names, buffer=intermediate_memory_buffer)
        if rag_vectordb_path:
            embedding_memory = EmbeddingSummaryMemory(llm=ChatOpenAI(temperature=0), memory_key='rag_log', input_key='input', output_key='intermediate_steps', filtered_tool_list=context_tool_names)
            
        memory = CombinedMemory(memories=[chat_memory, intermediate_memory, embedding_memory] if rag_vectordb_path else [chat_memory, intermediate_memory])
        return_intermediate_steps = True
    elif use_memory == 'both' and memory is None and agent_as_a_fsa:
        chat_memory = ConversationBufferWindowMemory(memory_key='chat_history', input_key='input', output_key='output')
        #TODO: add filtered list here for FSA
        intermediate_memory = FSAMemory(llm=ChatOpenAI(temperature=0, model='gpt-4'), fsa_object=fsa_object, memory_key='past_action_log', input_key='input', output_key='intermediate_steps', buffer=intermediate_memory_buffer)

        if rag_vectordb_path:
            embedding_memory = EmbeddingSummaryMemory(llm=ChatOpenAI(temperature=0), memory_key='rag_log', input_key='input', output_key='intermediate_steps', filtered_tool_list=context_tool_names)
        
        # Assign intermediate memory to the default fsa object if possible
        # TODO add validation here
        if fsa_object is not None and start_state is None and intermediate_memory_buffer == "":
            intermediate_memory.buffer = fsa_object().json(indent=2)
        elif fsa_object is None and start_state is not None and intermediate_memory_buffer == "":
            intermediate_memory.buffer = start_state.json(indent=2)

        memory = CombinedMemory(memories=[chat_memory, intermediate_memory, embedding_memory] if rag_vectordb_path else [chat_memory, intermediate_memory])

        return_intermediate_steps = True
    elif use_memory == 'chat' and memory is None:
        memory = ConversationBufferWindowMemory(memory_key='chat_history', input_key='input', output_key='output')
        return_intermediate_steps = False
    elif use_memory == 'action' and memory is None and not agent_as_a_fsa:
        memory = CustomActionLogSummaryMemory(llm=ChatOpenAI(temperature=0, model='gpt-4'), memory_key='past_action_log', input_key='input', output_key='intermediate_steps', filtered_tool_list=action_tool_names, buffer=intermediate_memory_buffer)
        return_intermediate_steps = True
    elif use_memory == 'action' and memory is None and agent_as_a_fsa:
        memory = FSAMemory(llm=ChatOpenAI(temperature=0, model='gpt-4'), fsa_object=fsa_object, memory_key='past_action_log', input_key='input', output_key='intermediate_steps', buffer=intermediate_memory_buffer)
        return_intermediate_steps = True
        if fsa_object is not None and start_state is None and intermediate_memory_buffer == "":
            memory.buffer = fsa_object().json(indent=2)
        elif fsa_object is None and start_state is not None and intermediate_memory_buffer == "":
            memory.buffer = start_state.json(indent=2)
    else:
        memory = None

    # Create agent from prompt and tools and return agent executor
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
        **agent_executor_kwargs
    )