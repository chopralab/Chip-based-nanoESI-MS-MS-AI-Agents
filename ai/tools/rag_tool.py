
#TODO: This is a part of future refactoring when the rag tools is used for abstraction.

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.chains import create_extraction_chain, RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor

#?QUESTION Do we assume that the FAISS is used to make the embeddings? or do we need to switch this to a different vector store if required?
@tool
def get_answer_from_information(situation: str, path_to_embeddings) -> str:
    """
    The user will provide a situation and path to the embeddings (documents).
    The tool will return the answer from the information by formulating a query from the situation and retrieving the answer from the documents.
    Provide a well formatted answer that is easy for the user to understand.
    """
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(path_to_embeddings, embeddings)
    retriever = db.as_retriever()
    return retriever.invoke(situation)

