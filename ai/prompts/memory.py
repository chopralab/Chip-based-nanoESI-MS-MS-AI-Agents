from langchain.prompts import PromptTemplate

ACTION_LOG_SUMMARY_TEMPLATE = """
You will receive a list of python dictionaries representing an AI agents internal though/action/observation process.

The first element of each tuple is the agent action, the second element is the observation of that action.

You should summarize the action and observation that the AI agent had for each element of this list.
You MUST include all values of each action's observation in your summary as they may be relevent later.

For example the following list of one agent thought/action/observation

Current Summary:
I have petted the cow and the cow says moo.

Human: Can you pet the dog?
AI:
[
    (AgentAction(
        tool=pet_dog, 
        tool_input={{}}, 
        log=Thought: To pet the dog, I need to use the pet_dog tool
        Action:
        {{   
            "action": "pet_dog",
            "action_input": {{}}
        }}
    ), 
        {{sound: woof, status: happy}} # This is the observation!
    )
]

New Summary:
I have petted the cow and the makes the moo sound. I have petted the dog, the dog make the woof sound and the dog's status is happy.

Current summary:
{summary}

New lines of conversation:
{new_lines}

New Summary:"""

ACTION_SUMMARY_PROMPT = PromptTemplate(
    input_variables=['summary', 'new_lines'],
    template=ACTION_LOG_SUMMARY_TEMPLATE
)

RAG_SUMMARY_TEMPLATE = """
You will receive a list of python dictionaries representing an AI agents internal though/action/observation process.

You should summarize the action and observation that the AI agent had for each element of this list.

Most important step is to Only summarize the information that is from the call_RAG_chain tool and ignore the rest. I want you to meantion all the things that were referred after the call_RAG_chain tool was used like the list of documents and their page numbers. This information is important for documentation and future reference.

For example the following list of one agent thought/action/observation

Current Summary:
Uptil now I knew that bunny rabbit has 4 legs and is a mammal.

Human: Can you tell me the amount of legs a dog has?
AI:
[
    (AgentAction(
        tool=call_RAG_chain, 
        tool_input={{"question": "What is the amount of legs a dog has?"}}, 
        log=Thought: This is not within my domain knowledge, I need to use the call_RAG_chain tool to query this information.
        Action:
        {{   
            "action": "call_RAG_chain",
            "action_input": {{"question" : "What is the number of legs a dog has?"}}
        }}
    ), 
        {{"output": "A dog has 4 legs."}}
    )
]

New Summary:
The AI just had information that bunny is a mammal and has 4 legs but it did not have the information about the amount of legs a dog has. The AI used the call_RAG_chain tool to query this information from relavant documentas and found out that a dog has 4 legs. It referred to the following list of documents dogs_info.pdf (page 5), mammals_info.pdf (page 10) and animals_info.pdf (page 15). Around 10 documents were referred to get this information.

Current summary:
{summary}

New lines of conversation:
{new_lines}

New Summary:"""

EMBEDDING_SUMMARY_PROMPT = PromptTemplate(
    input_variables=['summary', 'new_lines'],
    template = RAG_SUMMARY_TEMPLATE
)

ACTION_LOG_FSA_TEMPLATE = '''
Your goal is to update a dictionary which represent fields of a finite state automaton (FSA) based on new information.

You will receive a list of python dictionaries representing an AI agents internal though/action/observation process which controls this FSA.

View the formatting instructions for a description of each field and transition rules for that field. 

If transition rule(s) are provided, only update that field when agent actions fulfill those transition rule(s).

For example the following list of one agent thought/action/observation

Current FSA:
{{
    status: None
}}

Human: Can you get some information for me?
AI:
[
    (AgentAction(
        tool=get_status, 
        tool_input={{}}, 
        log=Thought: I need to get the status of the system
        Action:
        {{   
            "action": "get_status",
            "action_input": {{}}
        }}
    ), 
        {{status: busy'}}
    )
]

New FSA:
{{
    status: busy
}}

Current FSA:
{summary}

Internal Operations:
{new_lines}

Formatting Instructions
{formatting_instructions}

New FSA:'''