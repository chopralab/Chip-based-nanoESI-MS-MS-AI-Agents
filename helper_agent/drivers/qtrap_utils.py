"""
QTRAP Helper Agent Utilities
=============================
All utility functions for the QTRAP AI Agent system.

Usage:
    from qtrap_utils import setup_helper, tell_agent, ask_agent, run_helper_query
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from helper_agent_core import HelperAIAgent


# ================================================================================
# GLOBAL VARIABLES (will be set by setup functions)
# ================================================================================
_helper_instance = None
_embeddings = None
_doc_store = {}
_EMBEDdb = None


# ================================================================================
# DOCUMENT PROCESSING UTILITIES
# ================================================================================

def _mk_uid(d):
    """Create a unique ID for a document chunk."""
    src = str(d.metadata.get("source", "unknown"))
    pg = str(d.metadata.get("page", "-1"))
    h = hashlib.md5((d.page_content[:256] + src + pg).encode("utf-8")).hexdigest()[:12]
    return f"{os.path.basename(src)}#p{pg}#{h}"


def _mk_table_fixed(rows, headers):
    """Fix the table formatting bug in HelperAIAgent."""
    if not rows:
        return ""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = []
    for r in rows:
        row_cells = [(str(r.get(col, ""))).replace("\n", " ").strip() for col in headers]
        body_lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join([head, sep] + body_lines)


# ================================================================================
# RETRIEVER & LOADER FUNCTIONS
# ================================================================================

def my_retriever(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Retrieve documents from the FAISS index."""
    global _EMBEDdb
    if _EMBEDdb is None:
        raise RuntimeError("EMBEDdb not initialized. Call setup_faiss_index() first.")
    
    results = _EMBEDdb.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, score in results:
        uid = doc.metadata.get("uid", "")
        out.append({
            "id": uid,
            "score": float(score),
            "title": doc.metadata.get("title", "Untitled"),
            "url": "",
            "snippet": doc.page_content[:200]
        })
    return out


def my_loader(doc_id: str) -> Dict[str, Any]:
    """Load a document by ID from the document store."""
    global _doc_store
    return _doc_store.get(doc_id, {"id": doc_id, "title": "Unknown", "url": "", "text": ""})


# ================================================================================
# SETUP FUNCTIONS
# ================================================================================

def setup_faiss_index(
    base_path: str = "/home/sanjay/QTRAP_paper/Chip-based-nanoESI-MS-MS-AI-Agents/helper_agent",
    pdf_subdir: str = "recentlipids7",
    rebuild: bool = False
) -> tuple:
    """
    Setup or load the FAISS index and document store.
    
    Args:
        base_path: Base path to the helper_agent directory
        pdf_subdir: Subdirectory containing PDFs (relative to papers/)
        rebuild: If True, rebuild the index from scratch
    
    Returns:
        (embeddings, EMBEDdb, doc_store)
    """
    global _embeddings, _EMBEDdb, _doc_store
    
    target_dir = os.path.join(base_path, "papers")
    pdf_path = os.path.join(target_dir, pdf_subdir)
    faiss_index_path = os.path.join(target_dir, "faiss_index")
    docmap_path = os.path.join(faiss_index_path, "docmap.jsonl")
    
    # Initialize embeddings
    _embeddings = OpenAIEmbeddings()
    
    # Check if we need to rebuild
    if rebuild or not os.path.exists(faiss_index_path):
        print(f"Building FAISS index from {pdf_path}...")
        
        # Load PDFs
        os.chdir(target_dir)
        loader = DirectoryLoader(pdf_path, glob="**/*.pdf", loader_cls=PyPDFLoader,
                               show_progress=True, use_multithreading=True)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages.")
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks.")
        
        # Add UIDs
        for d in docs:
            d.metadata = dict(d.metadata)
            d.metadata["uid"] = _mk_uid(d)
            d.metadata["title"] = os.path.basename(d.metadata.get("source", "unknown"))
        
        # Save document map
        os.makedirs(faiss_index_path, exist_ok=True)
        with open(docmap_path, "w", encoding="utf-8") as f:
            for d in docs:
                rec = {
                    "id": d.metadata["uid"],
                    "title": d.metadata.get("title", "Untitled"),
                    "url": "",
                    "text": d.page_content
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved doc map: {docmap_path}")
        
        # Build FAISS index
        _EMBEDdb = FAISS.from_documents(docs, _embeddings)
        _EMBEDdb.save_local(faiss_index_path)
        print("Saved FAISS index to faiss_index/")
    else:
        print(f"Loading existing FAISS index from {faiss_index_path}...")
        _EMBEDdb = FAISS.load_local(faiss_index_path, _embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
    
    # Load document store
    _doc_store = {}
    with open(docmap_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            _doc_store[rec["id"]] = rec
    print(f"Loaded {len(_doc_store)} documents into memory.")
    
    return _embeddings, _EMBEDdb, _doc_store


def setup_helper(
    model: str = "gpt-4o",
    temperature: float = 0,
    base_path: str = "/home/sanjay/QTRAP_paper/Chip-based-nanoESI-MS-MS-AI-Agents/helper_agent",
    pdf_subdir: str = "recentlipids7",
    rebuild_index: bool = False
) -> HelperAIAgent:
    """
    Complete setup of the HelperAIAgent with FAISS index.
    
    Args:
        model: OpenAI model to use (default: gpt-4o)
        temperature: LLM temperature (default: 0)
        base_path: Base path to helper_agent directory
        pdf_subdir: PDF subdirectory name
        rebuild_index: Whether to rebuild the FAISS index
    
    Returns:
        Configured HelperAIAgent instance
    """
    global _helper_instance
    
    # Setup FAISS index
    setup_faiss_index(base_path, pdf_subdir, rebuild_index)
    
    # Configure LLM
    llm = ChatOpenAI(model=model, temperature=temperature)
    print(f"Configured LLM: {model}")
    
    # Patch the table bug
    HelperAIAgent._mk_table = staticmethod(_mk_table_fixed)
    print("Patched HelperAIAgent._mk_table")
    
    # Create helper instance
    _helper_instance = HelperAIAgent(retriever=my_retriever, loader=my_loader, llm=llm)
    print("Created HelperAIAgent instance - ready to use!")
    
    return _helper_instance


def get_helper() -> HelperAIAgent:
    """Get the global helper instance."""
    if _helper_instance is None:
        raise RuntimeError("Helper not initialized. Call setup_helper() first.")
    return _helper_instance


# ================================================================================
# QUERY FUNCTIONS
# ================================================================================

def run_helper_query(
    question: str,
    top_k: int = 8,
    max_tokens: int = 2000,
    log_dir: str = None
) -> Dict[str, Any]:
    """
    Run a query with full logging.
    
    Args:
        question: The question to ask
        top_k: Number of documents to retrieve
        max_tokens: Maximum tokens for context
        log_dir: Directory to save logs (default: logs/query/)
    
    Returns:
        Dict with answer, citations, and log file paths
    """
    helper = get_helper()
    
    # Use default organized directory structure
    if log_dir is None:
        log_base_dir = "/home/sanjay/QTRAP_paper/Chip-based-nanoESI-MS-MS-AI-Agents/helper_agent/logs"
        log_dir = os.path.join(log_base_dir, "query")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"query_log_{timestamp}.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Open log file
    with open(log_filepath, "w", encoding="utf-8") as log_file:
        # Write header
        header = f"{'='*80}\nAI Agent Query Log\n"
        header += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Question: {question}\n"
        header += f"Top-K: {top_k}, Max Tokens: {max_tokens}\n"
        header += f"{'='*80}\n\n"
        
        log_file.write(header)
        print(header)
        
        # Search and retrieve context
        search_msg = ">>> STEP 1: Searching corpus for relevant documents...\n"
        log_file.write(search_msg)
        print(search_msg, end="")
        
        sr = helper.search_corpus(question, top_k=top_k)
        ids = [r["id"] for r in sr["results"]]
        
        search_results = f"✓ Found {len(ids)} relevant documents\n\n"
        log_file.write(search_results)
        print(search_results)
        
        # Retrieve context
        retrieve_msg = ">>> STEP 2: Retrieving full context from documents...\n"
        log_file.write(retrieve_msg)
        print(retrieve_msg, end="")
        
        ctx = helper.retrieve_context(ids, max_tokens=max_tokens)
        
        context_msg = f"✓ Retrieved {len(ctx['context'])} characters of context\n\n"
        log_file.write(context_msg)
        print(context_msg)
        
        # Get the raw answer (with LLM call)
        answer_msg = ">>> STEP 3: Prompting AI Agent (LLM) to analyze context and answer...\n"
        log_file.write(answer_msg)
        print(answer_msg, end="")
        
        # Show that we're about to call the LLM
        llm_status = "🤖 Sending prompt to GPT-4o...\n"
        log_file.write(llm_status)
        print(llm_status, end="")
        
        rag_result = helper.answer_with_rag(question, ctx["context"], ctx["citations"])
        
        llm_complete = "✓ AI Agent response received\n\n"
        log_file.write(llm_complete)
        print(llm_complete)
        
        # Format output
        answer_section = f"\n{'='*80}\n=== AI AGENT ANSWER ===\n{'='*80}\n"
        answer_section += f"{rag_result['answer']}\n\n"
        
        citations_section = f"{'='*80}\n=== CITATIONS ===\n{'='*80}\n"
        for i, cit in enumerate(rag_result.get("citations", []), 1):
            citations_section += f"{i}. {cit.get('title', 'Unknown')}\n"
        
        # Write full output to log
        full_output = answer_section + citations_section + f"\n{'='*80}\n"
        full_output += f"Log saved to: {log_filepath}\n"
        full_output += f"{'='*80}\n"
        
        log_file.write(full_output)
        print(full_output)
        
        # Also save JSON version
        json_filename = f"query_log_{timestamp}.json"
        json_filepath = os.path.join(log_dir, json_filename)
        
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "answer": rag_result["answer"],
            "citations": rag_result.get("citations", []),
            "search_results_count": len(ids),
            "context_length": len(ctx['context']),
            "llm_model": "gpt-4o" if helper.llm else "none"
        }
        
        with open(json_filepath, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2, ensure_ascii=False)
        
        print(f"JSON log saved to: {json_filepath}\n")
    
    # Return just answer and citations (no wrapper)
    return {
        "answer": rag_result["answer"],
        "citations": rag_result["citations"],
        "log_file": log_filepath,
        "json_file": json_filepath
    }


# ================================================================================
# MEMORY FUNCTIONS
# ================================================================================

def tell_agent(observation: str, save_log: bool = True) -> str:
    """
    Tell the agent something to remember.
    
    Args:
        observation: The text to remember
        save_log: Whether to save a log file (default: True)
    
    Returns:
        The memory key
    """
    helper = get_helper()
    
    key = f"obs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    data = {
        "text": observation,
        "timestamp": datetime.now().isoformat()
    }
    
    helper.upsert_memory(
        namespace="conversations",
        key=key,
        value_json=json.dumps(data),
        ttl_days=0
    )
    
    print(f"✅ Remembered: {observation[:100]}...")
    
    # Save log if enabled
    if save_log:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "action": "tell_agent",
            "memory_key": key,
            "observation": observation,
            "observation_length": len(observation),
            "namespace": "conversations"
        }
        _save_tell_agent_log(log_data)
    
    return key


def ask_agent(question: str, top_k: int = 8, save_log: bool = True) -> str:
    """
    Ask the agent - prioritizes memory over RAG with LLM reasoning.
    
    Args:
        question: The question to ask
        top_k: Number of documents to retrieve if no memory exists
        save_log: Whether to save a log file (default: True)
    
    Returns:
        The answer string
    """
    helper = get_helper()
    
    # Setup logging if enabled
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "top_k": top_k,
        "method": None,
        "answer": None,
        "memory_used": False,
        "memory_count": 0,
        "citations": []
    }
    
    # Get all memories
    mem_list = helper.list_memory("conversations")
    memory_entries = []
    
    if mem_list.get("keys"):
        for key in mem_list["keys"]:
            mem = helper.get_memory("conversations", key)
            if mem.get("found"):
                memory_entries.append(mem['value']['text'])
    
    log_data["memory_count"] = len(memory_entries)
    
    # If we have memories, use them as the PRIMARY context
    if memory_entries:
        log_data["memory_used"] = True
        log_data["method"] = "memory"
        memory_text = "\n\n".join(memory_entries)
        
        print("🧠 Using stored experimental data...")
        
        # Use LLM to reason about the memory
        llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        
        prompt = f"""Based on the following validated experimental results, answer the user's question.

VALIDATED EXPERIMENTAL RESULTS:
{memory_text}

USER QUESTION: {question}

Provide a clear, synthesized answer that:
1. Directly addresses the question
2. References the key findings from the experimental data
3. Provides the specific recommendation
4. Explains WHY this recommendation is better

Answer:"""
        
        response = llm.invoke(prompt)
        answer = response.content
        log_data["answer"] = answer
        
        print(answer)
        print("\n📌 Source: Lab Validated Experimental Results")
        
        # Save log if enabled
        if save_log:
            _save_ask_agent_log(log_data)
        
        return answer
    
    else:
        # No memory - fall back to RAG
        log_data["method"] = "rag"
        print("📚 No stored data found, searching literature...")
        sr = helper.search_corpus(question, top_k=top_k)
        ids = [r["id"] for r in sr["results"]]
        ctx = helper.retrieve_context(ids, max_tokens=2000)
        
        rag_result = helper.answer_with_rag(question, ctx["context"], ctx["citations"])
        log_data["answer"] = rag_result["answer"]
        log_data["citations"] = rag_result.get("citations", [])
        
        print(rag_result["answer"])
        print("\n📌 Source: Published Literature")
        
        # Save log if enabled
        if save_log:
            _save_ask_agent_log(log_data)
        
        return rag_result["answer"]


def _save_ask_agent_log(log_data: Dict[str, Any]) -> None:
    """Internal function to save ask_agent logs."""
    log_base_dir = "/home/sanjay/QTRAP_paper/Chip-based-nanoESI-MS-MS-AI-Agents/helper_agent/logs"
    log_dir = os.path.join(log_base_dir, "ask_agent")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filepath = os.path.join(log_dir, f"ask_agent_log_{timestamp}.json")
    txt_filepath = os.path.join(log_dir, f"ask_agent_log_{timestamp}.txt")
    
    # Save JSON
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # Save text version
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("ASK AGENT LOG (Memory-First Query)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {log_data['timestamp']}\n")
        f.write(f"Question: {log_data['question']}\n")
        f.write(f"Method: {log_data['method']}\n")
        f.write(f"Memory Used: {log_data['memory_used']}\n")
        f.write(f"Memory Entries: {log_data['memory_count']}\n\n")
        f.write("="*80 + "\n")
        f.write("ANSWER\n")
        f.write("="*80 + "\n")
        f.write(f"{log_data['answer']}\n\n")
        
        if log_data['citations']:
            f.write("="*80 + "\n")
            f.write("CITATIONS\n")
            f.write("="*80 + "\n")
            for i, cit in enumerate(log_data['citations'], 1):
                f.write(f"{i}. {cit.get('title', 'Unknown')}\n")
    
    print(f"\n💾 Log saved: {json_filepath}")
    print(f"💾 Log saved: {txt_filepath}")


def _save_tell_agent_log(log_data: Dict[str, Any]) -> None:
    """Internal function to save tell_agent logs."""
    log_base_dir = "/home/sanjay/QTRAP_paper/Chip-based-nanoESI-MS-MS-AI-Agents/helper_agent/logs"
    log_dir = os.path.join(log_base_dir, "tell_agent")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filepath = os.path.join(log_dir, f"tell_agent_log_{timestamp}.json")
    txt_filepath = os.path.join(log_dir, f"tell_agent_log_{timestamp}.txt")
    
    # Save JSON
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # Save text version
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("TELL AGENT LOG (Memory Storage)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {log_data['timestamp']}\n")
        f.write(f"Action: {log_data['action']}\n")
        f.write(f"Memory Key: {log_data['memory_key']}\n")
        f.write(f"Namespace: {log_data['namespace']}\n")
        f.write(f"Observation Length: {log_data['observation_length']} characters\n\n")
        f.write("="*80 + "\n")
        f.write("STORED OBSERVATION\n")
        f.write("="*80 + "\n")
        f.write(f"{log_data['observation']}\n\n")
        f.write("="*80 + "\n")
        f.write(f"✅ Memory stored successfully with key: {log_data['memory_key']}\n")
        f.write("="*80 + "\n")
    
    print(f"\n💾 Log saved: {json_filepath}")
    print(f"💾 Log saved: {txt_filepath}")


def clear_memory(namespace: str = "conversations") -> None:
    """Clear all memories in a namespace."""
    helper = get_helper()
    mem_list = helper.list_memory(namespace)
    
    count = 0
    if mem_list.get("keys"):
        for key in mem_list["keys"]:
            helper.delete_memory(namespace, key)
            count += 1
    
    print(f"🗑️  Cleared {count} memories from '{namespace}' namespace")


def list_memories(namespace: str = "conversations") -> List[Dict[str, Any]]:
    """List all memories in a namespace."""
    helper = get_helper()
    mem_list = helper.list_memory(namespace)
    
    memories = []
    if mem_list.get("keys"):
        for key in mem_list["keys"]:
            mem = helper.get_memory(namespace, key)
            if mem.get("found"):
                memories.append({
                    "key": key,
                    "text": mem['value']['text'],
                    "timestamp": mem['value'].get('timestamp', 'unknown')
                })
    
    return memories
