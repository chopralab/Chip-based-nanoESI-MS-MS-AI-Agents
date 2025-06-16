import os
from typing import Any, Dict, List, Optional, TypedDict, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# -------------------------------
# Minimal State Definition
# -------------------------------
class HelperAgentState(TypedDict):
    messages: List[BaseMessage]
    agent_response: Optional[str]
    agent_state: Dict[str, Any]

# -------------------------------
# Minimal Node Function
# -------------------------------
def helper_agent_node(state: HelperAgentState, config=None) -> HelperAgentState:
    # --- Dummy PDF RAG for testing ---
    PDF_PATH = "sample.pdf"  # Place a PDF named sample.pdf in your cwd for test
    
    if os.path.exists(PDF_PATH):
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        # Just a demo search, not used further in this minimal example
        _ = db.similarity_search("QTRAP")
    else:
        docs = []
    
    # Pick up user message
    last_user_message = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_message = m.content
            break
    if last_user_message is None:
        last_user_message = "Dummy test question?"

    # Simple canned response
    response = f"[TEST RESPONSE] Received: {last_user_message}"
    return {
        **state,
        "agent_response": response,
        "messages": state["messages"] + [AIMessage(content=response)],
        "agent_state": {"test": True}
    }

def route_end(state: HelperAgentState) -> Literal["__end__"]:
    return "__end__"

# -------------------------------
# Minimal Graph Construction
# -------------------------------
builder = StateGraph(HelperAgentState)
builder.add_node("helper_agent_node", helper_agent_node)
builder.set_entry_point("helper_agent_node")
builder.add_conditional_edges("helper_agent_node", route_end)
graph = builder.compile()

# -------------------------------
# Minimal Run
# -------------------------------
if __name__ == "__main__":
    initial_state: HelperAgentState = {
        "messages": [HumanMessage(content="Is this working?")],
        "agent_response": None,
        "agent_state": {},
    }
    output = graph.invoke(initial_state)
    print(output["agent_response"])
