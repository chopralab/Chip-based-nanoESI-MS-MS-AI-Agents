import json
from typing import Any, Literal, Type
from pydantic import BaseModel as BaseModelV2

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.memory import BaseMemory
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from langchain.tools.human import HumanInputRun
from langchain.memory import (
    ConversationBufferWindowMemory,
    CombinedMemory
)
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
#we will rename this to sciborg tool (or some other name)
from sciborg_dev.ai.tools.core import LinqxTool
from sciborg_dev.ai.agents.rag_agent import rag_agent


class SciborgAgent:
    def __init__(
        self,
        microservice: BaseDriverMicroservice,
        llm: BaseLanguageModel = ChatOpenAI(temperature=0, model='gpt-4'),
        prompt_template: str = BASE_LINQX_CHAT_PROMPT_TEMPLATE,
        use_memory: Literal['chat', 'action', 'embedding', 'all'] | None = None,
        memory: BaseMemory | None = None,
        intermediate_memory_buffer: str = "",
        past_action_log: str = "",
        human_interaction: bool = False,
        assume_defaults: bool = False,
        rag_vectordb_path: str | None = None,
        agent_description: str | None = None,
        agent_as_a_tool: AgentExecutor | None = None,
        agent_as_a_fsa: bool = False,
        fsa_object: Type[BaseModelV2] | None = None,
        start_state: BaseModelV2 | None = None,
        use_linqx_tools: bool = True,
        handle_tool_error: bool = True,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        memory_file: str = None,
        **agent_executor_kwargs
    ):
        #assign the parameters to the class variables
        self.microservice = microservice
        self.rag_vectordb_path = rag_vectordb_path
        self.llm = llm
        self.use_memory = use_memory
        self.memory = memory
        self.human_interaction = human_interaction
        self.assume_defaults = assume_defaults
        self.verbose = verbose
        self.return_intermediate_steps = return_intermediate_steps
        self.agent_executor_kwargs = agent_executor_kwargs
        self.memory_file = memory_file
        self.input_variables = None

        #do the internal initializations
        self.tools = self._build_tools(use_linqx_tools, handle_tool_error, agent_as_a_tool, agent_description)
        self.memory = self._initialize_memory(intermediate_memory_buffer, past_action_log, agent_as_a_fsa, fsa_object, start_state)
        self.prompt = self._build_prompt(prompt_template , rag_vectordb_path, past_action_log)
        self.agent_executor = self._create_agent_executor()
        
        
    
    def _build_tools(self, use_linqx_tools, handle_tool_error, agent_as_a_tool, agent_description):
        tools = [LinqxTool(linqx_command=command, handle_tool_error=handle_tool_error) for command in self.microservice.commands.values()] if use_linqx_tools else [self._command_to_tool(command) for command in self.microservice.commands.values()]
        if self.human_interaction:
            tools.append(HumanInputRun())
        if self.rag_vectordb_path:
            @tool
            def call_RAG_agent(question: str) -> str:
                """
                    This is the function that will query the relevant sources of information to get the answer to the question.
                    There will be situations when you are not able to answer the question directly from the information you currently have. In such cases, you can search for the answer in the relevant sources of information.
                    Often the user will also specify that you need to refer to "information" or "documents" to get the answer.
                    TAKSK: You have to frame the best possible "question" that is extremely descriptive and then use it as a parameter to query the relevant sources of information and return the citations if present.
                """
                RAG_agent = rag_agent(question, self.rag_vectordb_path)
                return RAG_agent.invoke({"question": question})['output']
            tools.append(call_RAG_agent)
        if agent_as_a_tool:
            @tool
            def call_provided_Agent(question: str) -> str:
                """
                This is the function that will call the provided agent as a tool.
                """
                output = agent_as_a_tool.invoke({"input": question})
                return output['output']
            tools.append(call_provided_Agent)
        return tools

    def _initialize_memory(self, intermediate_memory_buffer, past_action_log, agent_as_a_fsa, fsa_object, start_state):
        action_tool_names = [tool.name for tool in self.tools]
        use_memory = self.use_memory
        memories = []

        rag_vectordb_path = self.rag_vectordb_path

        if isinstance(use_memory, list):
            # Handle multiple memories
            for memory_type in use_memory:
                if memory_type == 'chat':
                    memories.append(ConversationBufferWindowMemory(
                        memory_key='chat_history',
                        input_key='input',
                        output_key='output'
                    ))
                if memory_type == 'action':
                    memories.append(CustomActionLogSummaryMemory(
                        llm=ChatOpenAI(temperature=0, model='gpt-4'),
                        memory_key='past_action_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        buffer=intermediate_memory_buffer,
                        filtered_tool_list=action_tool_names
                    ))
                if memory_type == 'embedding' and rag_vectordb_path:
                    memories.append(EmbeddingSummaryMemory(
                        llm=ChatOpenAI(temperature=0),
                        memory_key='rag_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        filtered_tool_list=action_tool_names
                    ))
                #TODO: complete the FSA integration in the memory
                if memory_type == 'fsa' and agent_as_a_fsa:
                    memories.append(FSAMemory(
                        llm=ChatOpenAI(temperature=0, model='gpt-4'),
                        memory_key='fsa_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        fsa_object=fsa_object,
                        buffer=intermediate_memory_buffer,
                    ))
                
            return CombinedMemory(memories=memories) if memories else None

        # Handle single memory type
        elif use_memory:
            if use_memory == 'chat':
                return ConversationBufferWindowMemory(
                    memory_key='chat_history',
                    input_key='input',
                    output_key='output'
                )
            if use_memory == 'action':
                return CustomActionLogSummaryMemory(
                    llm=ChatOpenAI(temperature=0, model='gpt-4'),
                    memory_key='past_action_log',
                    input_key='input',
                    output_key='intermediate_steps',
                    buffer=intermediate_memory_buffer,
                    filtered_tool_list=action_tool_names
                )
            if use_memory == 'embedding' and rag_vectordb_path:
                return EmbeddingSummaryMemory(
                    llm=ChatOpenAI(temperature=0),
                    memory_key='rag_log',
                    input_key='input',
                    output_key='intermediate_steps'
                )
            if use_memory == 'fsa' and agent_as_a_fsa:
                return FSAMemory(
                    llm=ChatOpenAI(temperature=0, model='gpt-4'),
                    memory_key='fsa_log',
                    input_key='input',
                    output_key='intermediate_steps',
                    fsa_object=fsa_object,
                    buffer=intermediate_memory_buffer,
                )
            if use_memory == 'all':
                memories = [
                    ConversationBufferWindowMemory(
                        memory_key='chat_history',
                        input_key='input',
                        output_key='output'
                    ),
                    CustomActionLogSummaryMemory(
                        llm=self.llm,
                        memory_key='past_action_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        buffer=intermediate_memory_buffer,
                        filtered_tool_list=action_tool_names
                    )
                ]
                if self.rag_vectordb_path:
                    memories.append(EmbeddingSummaryMemory(
                        llm=self.llm,
                        memory_key='rag_log',
                        input_key='input',
                        output_key='intermediate_steps'
                    ))
                if agent_as_a_fsa:
                    memories.append(FSAMemory(
                        llm=self.llm,
                        memory_key='fsa_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        fsa_object=fsa_object,
                        buffer=intermediate_memory_buffer,
                    ))
                return CombinedMemory(memories=memories)
        
        return None

    def _build_prompt(self, prompt_template, rag_vectordb_path, past_action_log):
        input_variables = ['tools', 'tool_names', 'agent_scratchpad']
        #TODO: not all the input variables are presen in here, refer to the old implementaiton and fix this
        partial_variables = {
            'microservice': self.microservice.name,
            'microservice_description': self.microservice.desc,
        }

        # we need to account for all combinations
        # use_memory can be a list of strings or a single string
        use_memory = self.use_memory
        #TODO: discuss with Matt about the memory implemenetation
        if isinstance(use_memory, list):
            #TODO: we need to add in validation checks for memory types
            #there are multiple memories to be used
            #we will have to check for each memory type and then add the required variables to the input_variables
            for memory_type in use_memory:
                self.return_intermediate_steps = True
                if memory_type == 'chat':
                    input_variables.append('chat_history')
                if memory_type == 'action':
                    input_variables.append('past_action_log')
                if memory_type == 'embedding' and rag_vectordb_path:
                    input_variables.append('embedding_log')

        #TODO: this is redundant but done to preserve the old implementation
        elif use_memory:
            #TODO: Why do we need to fore this to be true, to save everything in memory right? so why not just remove it as a parameter and set it to true whenever user needs memory to be used
            self.return_intermediate_steps = True
            if use_memory == 'chat':
                input_variables.append('chat_history')
            if use_memory == 'action':
                input_variables.append('past_action_log')
            if use_memory == 'embedding' and rag_vectordb_path:
                input_variables.append('embedding_log')
            if use_memory == 'all':
                input_variables.append('chat_history')
                input_variables.append('past_action_log')
                if rag_vectordb_path:
                    input_variables.append('embedding_log')

        # Adding instructions
        TOTAL_INSTRUCTIONS = ""
        if self.human_interaction:
            TOTAL_INSTRUCTIONS += HUMAN_TOOL_INSTRUCTIONS
        if self.assume_defaults:
            TOTAL_INSTRUCTIONS += ASSUME_DEFAULTS_INSTRUCTIONS
        if self.rag_vectordb_path:
            TOTAL_INSTRUCTIONS += RAG_AS_A_TOOL_INSTRUCTIONS
        
        partial_variables['additional_instructions'] = TOTAL_INSTRUCTIONS

        return PromptTemplate(
            input_variables=input_variables,
            partial_variables=partial_variables,
            template=prompt_template
        )
    
    def _create_agent_executor(self):
        agent = create_structured_chat_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)

        agext_exec = AgentExecutor(agent=agent, tools=self.tools, memory=self.memory, verbose=self.verbose, return_intermediate_steps=self.return_intermediate_steps, **self.agent_executor_kwargs)
        if self.memory_file:
            memories = self._load_memory()
            agext_exec.memory.memories = memories
        return agext_exec
    
    #TODO: we need to assign the default name for the memory file dynamically as each execution should have a different memory file, also provide an option to save the memory file with a different name
    def save_memory(self, memory_file=None):
        if not memory_file:
            print("Memory file name not provided, saving memory to default file called memory.json")
            self.memory_file = "memory.json"
        else:
            self.memory_file = memory_file
            print(f"Memory file name provided, saving memory to {memory_file}")

        memories_data = {}

        #TODO: We need to discuss memory properly as we are currently not saing the paths an agent takes to reach a certain state, we are only saving the final state of the memory, for exploration agent the paths will be extermely important

        #TODO: we need to look into saving the memory object according to the llm used
        #add in a variable to save the llm name and model - also read this in loading and then instantiate the llm object accordingly

        for i, memory in enumerate(self.agent_executor.memory.memories):
            if memory:
                #check the type of memory and then save it accordingly
                if isinstance(memory, ConversationBufferWindowMemory):
                    #for this memory we will append/load the chat_memory data to the new memory object
        
                    data_to_add = []
                    for message in memory.buffer_as_messages:
                        if isinstance(message, HumanMessage):
                            message_type = "human"
                        elif isinstance(message, AIMessage):
                            message_type = "ai"
                        else:
                            message_type = "unknown"
                        message_data = {
                            "message_type": message_type,
                            "content": message.content,
                            #TODO: we can include additional_kwargs or response_metadata here but we need to see its importance
                        }
                        data_to_add.append(message_data)

                    memories_data['ConversationBufferWindowMemory'] = {
                        "memory_key": memory.memory_key,
                        "input_key": memory.input_key,
                        "output_key": memory.output_key,
                        "messages": data_to_add
                    }

                if isinstance(memory, CustomActionLogSummaryMemory):

                    memories_data['CustomActionLogSummaryMemory'] = {
                        "memory_key": memory.memory_key,
                        "input_key": memory.input_key,
                        "output_key": memory.output_key,
                        "buffer": memory.buffer,
                        "filtered_tool_list": memory.filtered_tool_list
                    }

                #TODO: add in the support for FSA memory after talking with Matt
                # if isinstance(memory, FSAMemory):

                #     if "FSAMemory" not in memory_from_file:
                #         memories_data['FSAMemory'] = {
                #             "memory_key": memory.memory_key,
                #             "input_key": memory.input_key,
                #             "output_key": memory.output_key,
                #             "fsa_object": memory.fsa_object,
                #             "start_state": memory.start_state
                #         }


                if isinstance(memory, EmbeddingSummaryMemory):

                    memories_data['EmbeddingSummaryMemory'] = {
                        "memory_key": memory.memory_key,
                        "input_key": memory.input_key,
                        "output_key": memory.output_key,
                        "buffer": memory.buffer,
                        "filtered_tool_list": memory.filtered_tool_list
                    }

        with open(self.memory_file, "w") as f:
            json.dump(memories_data, f, indent=4)

        print("Memory saved successfully")
        #print the location of the memory file
        print(f"Memory file saved at {self.memory_file}")

    def _load_memory(self):

        #load the memory file and then load the data into the memory object
        memory_from_file = {}
        try:
            with open(self.memory_file, "r") as f:
                memories_data = json.load(f)
        except FileNotFoundError:
            pass

        memories = []

        for memory_name, memory_data in memories_data.items():
            if memory_name == 'ConversationBufferWindowMemory':

                #we will be parsing the memory data in the agent call and then passing it as previous context before the current question

                add_messages = []
                for message_data in memory_data['messages']:
                    if message_data['message_type'] == "human":
                        message = HumanMessage(content=message_data['content'])
                    elif message_data['message_type'] == "ai":
                        message = AIMessage(content=message_data['content'])
                    else:
                        message = None
                    add_messages.append(message)

                to_add = ConversationBufferWindowMemory(
                    memory_key=memory_data['memory_key'],
                    input_key=memory_data['input_key'],
                    output_key=memory_data['output_key']
                )
                #We are adding the messages in the memory
                to_add.chat_memory.messages = add_messages
                #? The to_add.buffer , to_add.buffer_as_messages, to_add.buffer_as_strings are all read only properties, so we cannot assign them directly
                #? We are able to set the to_add.chat_memory parameter to message_history
                memories.append(to_add)
            
            if memory_name == 'CustomActionLogSummaryMemory':
                to_add = CustomActionLogSummaryMemory(
                    llm=ChatOpenAI(temperature=0, model='gpt-4'),
                    memory_key=memory_data['memory_key'],
                    input_key=memory_data['input_key'],
                    output_key=memory_data['output_key'],
                    buffer=memory_data['buffer'],
                    filtered_tool_list=memory_data['filtered_tool_list']
                )
                memories.append(to_add)

            if memory_name == 'EmbeddingSummaryMemory':
                self.memory = EmbeddingSummaryMemory(
                    llm=ChatOpenAI(temperature=0, model='gpt-4'),
                    memory_key=memory_data['memory_key'],
                    input_key=memory_data['input_key'],
                    output_key=memory_data['output_key'],
                    buffer=memory_data['buffer'],
                    filtered_tool_list=memory_data['filtered_tool_list']
                )
                memories.append(to_add)
        
        return memories
    
    def invoke(self, input_text) -> Any:
        if isinstance(input_text, dict):
            input_text = input_text.get("input", "")
        
        return self.agent_executor.invoke({"input": input_text})

#TODO: also change the way memory is instantiated in the agent class as we do not need a both option currently, we can do a seperate if statement for each requested memory type and then instantiate the memory object accordingly

#TODO: We need to change the base prompt instructions to refer to memory when required, even after chan
#TODO: this is not working for the fsa memory and we need to see if the code done for the core.py file matches the core2.py file
#TODO: Even after changing the prompt the agent does not first check the memory for the previous context (Why is this?)