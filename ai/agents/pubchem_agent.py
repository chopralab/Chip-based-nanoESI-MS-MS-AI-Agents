
from langchain_openai import ChatOpenAI
from langchain.agents import tool

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor

from sciborg.ai.tools.pubchem_tools import get_sids_from_cid, get_cids_from_sid, get_synonym, get_description, get_assay_description, get_assay_id_from_smiles, get_assay_name_from_aid, get_compound_property_table

def pubchem_agent(
        question: str,
        llm: BaseLanguageModel = ChatOpenAI(temperature=0.1)
        ): #?QUESTION Is this a good default value?)
    """
    Create a PubChem agent with the tools. This agent will be able to answer questions related to Chemistry using PubChem API. Whenever you want to look up any information related to Chemistry, you can use this agent over your knowledge base.
    """

    tools = [get_sids_from_cid, get_cids_from_sid, get_synonym, get_description, get_assay_description, get_assay_id_from_smiles, get_assay_name_from_aid, get_compound_property_table]

    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are very powerful assistant. 
                Your task is to use the appropriate tool to give what the user is asking for.
                    1. Rephrase the question in a query in the most explanatory way possible.
                    2. Break the task down into smaller parts depending on the tools that are available to you.
                    2. Call the tools one by one that are available to you to get the result.  
                    3. Provide the answer in a clear format to the user.

                Valid input types: sid: substance id,
                cid: compound id,
                aid: assay id
                smiles: The simplified molecular-input line-entry system is a specification in the form of a line notation for describing the structure of chemical species using short ASCII strings. E.g.: C=C, Na.Cl, C#N, CBr, CC.
                names: These are common names of chemical compounds 

                In your final response, do not add any urls or hyperlinks
                """, 
            ),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    chain = (
        {
            "question": lambda x: x["question"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    pubchem_agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True)

    return pubchem_agent_executor