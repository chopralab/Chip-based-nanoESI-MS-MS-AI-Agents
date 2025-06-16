import os

from typing import Any, Dict, List, Optional, TypedDict, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# ----------- Configurable Paths ------------
PDF_DIR = "/home/qtrap/sciborg_dev/notebooks/papers/qtrap_nano"
VECTOR_DB_PATH = "/home/qtrap/sciborg_dev/faiss_index_qtrap_nano"

print(f"Current working directory: {os.getcwd()}")
print(f"VECTOR_DB_PATH: {VECTOR_DB_PATH if 'VECTOR_DB_PATH' in globals() else 'Not defined yet'}")
print(f"PDF_DIR: {PDF_DIR if 'PDF_DIR' in globals() else 'Not defined yet'}")
# ----------- State -------------
class HelperAgentState(TypedDict):
    messages: List[BaseMessage]
    agent_response: Optional[str]
    agent_state: Dict[str, Any]

def helper_agent_node(state: HelperAgentState, config=None) -> HelperAgentState:
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    print(f"Trying to load FAISS index from: {VECTOR_DB_PATH}")
    print(f"Exists? {os.path.exists(VECTOR_DB_PATH)}")
    if os.path.exists(VECTOR_DB_PATH):
        print("Contents:", os.listdir(VECTOR_DB_PATH))
    else:
        print("FAISS index directory does NOT exist!")
    vectordb = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    last_user_message = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_message = m.content
            break
    if last_user_message is None:
        last_user_message = "What are these papers about?"

    retrieved_docs = vectordb.similarity_search(last_user_message, k=4)
    print("Retrieved the following chunks:")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Chunk {i} ---")
        print(doc.page_content[:500])  # Print the first 500 characters
        print()
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = (
        "You are a helpful assistant. Use the following context from scientific papers to answer the user's question.\n"
        "If the answer isn't in the papers, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {last_user_message}\nAnswer:"
    )
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key="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")
    response = llm.invoke(prompt).content

    return {
        **state,
        "agent_response": response,
        "messages": state["messages"] + [AIMessage(content=response)],
        "agent_state": {},
    }


def route_end(state: HelperAgentState) -> Literal["__end__"]:
    return "__end__"

# ----------- Graph Setup -------------
builder = StateGraph(HelperAgentState)
builder.add_node("helper_agent_node", helper_agent_node)
builder.set_entry_point("helper_agent_node")
builder.add_conditional_edges("helper_agent_node", route_end)
graph = builder.compile()

# ----------- Run (Interactive CLI) -------------
if __name__ == "__main__":
    initial_state: HelperAgentState = {
        "messages": [],
        "agent_response": None,
        "agent_state": {},
    }
    print("Ask a question about the papers (type 'exit' to quit):")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        initial_state["messages"].append(HumanMessage(content=user_input))
        output = graph.invoke(initial_state)
        print("\nAgent:", output["agent_response"])
        initial_state = output  # update state to keep conversation context
