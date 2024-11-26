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

def rag_agent(
    question: str,
    path_to_embeddings: str,
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1) #?QUESTION Is this a good default value?
):
    """
    Create a RAG chain with the tools.
    """
    embeddings = OpenAIEmbeddings()
    # # #?QUESTION Do we assume that the FAISS is used to make the embeddings? or do we need to switch this to a different vector store if required?
    db = FAISS.load_local(path_to_embeddings, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    @tool
    def get_answer_from_information(situation: str) -> str:
        """
        The user will provide a situation and the tool will return the answer from the information.
        Provide a well formatted answer that is easy for the user to understand.
        """
        results = retriever.invoke(situation)
        answers = []
        for result in results:
            answer = {}
            # result.page_content, result.metadata['source'], result.metadata['page']
            answer['citation_title'] = result.metadata['source']
            answer['citation_page'] = result.metadata['page']
            answer['answer'] = result.page_content
            answers.append(answer)
        return answers

    tools = [get_answer_from_information]

    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are very powerful assistant. 
                You can answer questions about user query using relevant sources. You have to provide factual answers.
                If requested to provide answer to the question you may use following steps:
                    1. First use the tool to retrieve relevant information
                    2. Analyze the retrieved information carefully
                    3. Provide a clear answer based ONLY on the retrieved information.
                    4. Return the answer with respective citations (title and page) present in the retrieved information.
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

    rag_agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True)

    return rag_agent_executor