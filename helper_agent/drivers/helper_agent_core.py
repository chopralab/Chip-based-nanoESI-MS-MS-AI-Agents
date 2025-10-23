"""
HelperAIAgent: Retrieval-Augmented Generation (RAG) + JSON-file memory + Output Formatter

Purpose
-------
A domain-agnostic helper module that:
  1) Searches your indexed corpora (RAG) and answers with citations.
  2) Persists durable context using a lightweight JSON memory store.
  3) Formats noisy agent runs into clean Markdown + JSON summaries.

Grounding & Non-Fabrication
---------------------------
- Always *observe results first* (retriever/loader-provided content) and
  generate answers grounded strictly in that context.
- If there is insufficient context to answer, *abstain* and say what’s missing.
- Do NOT make up facts, citations, or metadata.

Integration Points
------------------
- Provide a `retriever` callable to `search_corpus` via constructor. It should
  accept (query:str, top_k:int, filters:dict|None) and return a list of:
      { "id": str, "score": float, "title": str, "url": Optional[str], "snippet": str }
- Provide a `loader` callable to `retrieve_context` via constructor. It should
  accept (doc_id:str) and return:
      { "id": str, "title": str, "url": Optional[str], "text": str }

Public API
----------
RAG:
- search_corpus(query:str, top_k:int=10, filters_json:str="{}") -> dict
- retrieve_context(document_ids: list[str], max_tokens:int=3000) -> dict
- answer_with_rag(question:str, context:str, citations:list[dict]) -> dict
  (returns answer + auto-generated report_markdown/report_json)

Memory:
- upsert_memory(namespace:str, key:str, value_json:str, ttl_days:int=0) -> dict
- get_memory(namespace:str, key:str) -> dict
- delete_memory(namespace:str, key:str) -> dict
- list_memory(namespace:str, prefix:str="", limit:int=100) -> dict
- load_memory_file(file_path:str) -> dict
- save_memory_file(file_path:str, namespaces:list[str]|None=None) -> dict

Formatter:
- format_run_report(run: dict) -> {"markdown": str, "summary_json": dict}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import json
import re


# === Auto-knowledge helpers (top-level) ===
from collections import defaultdict
import hashlib, re, json as _json

def _norm(s: str) -> str:
    return " ".join(str(s).strip().split())

def _claim_id(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:12]

def _confidence_from_retrieval(citations, search_results=None):
    n_src = len({(c.get("id") or c.get("url") or "") for c in (citations or []) if (c.get("id") or c.get("url"))})
    base = min(1.0, 0.3 + 0.15 * n_src)  # ~0.45 for 1 src, ~0.9 for 4 srcs
    return round(base, 2)

def _canonicalize_value(v: str) -> str:
    return _norm(v).replace(" (v/v)", "").replace(" v/v", "")

def _kv_from_line(line: str):
    # Try to match the main pattern
    m = re.search(r'(?i)([A-Za-z0-9+\-/ :,]+)\s*\((\d+\s*:\s*\d+(?::\d+)?)\s*,?\s*v/(?:v|v/v)?\)', line)
    if m:
        k = _norm(m.group(1)).lower()
        v = _canonicalize_value(m.group(2))
        return k, v
    # Fallback: If the line contains 'v/v' and a ratio, try to extract more context
    m2 = re.search(r'([A-Za-z0-9+\-/ :,]+):\s*(\d+\s*:\s*\d+(?::\d+)?).*v/v', line)
    if m2:
        k = _norm(m2.group(1)).lower()
        v = _canonicalize_value(m2.group(2))
        return k, v
    return None, None

def _collect_candidate_claims(context: str):
    claims = []
    # Join lines to avoid truncation due to line breaks
    joined_context = ' '.join((context or '').splitlines())
    # Split into candidate sentences for extraction
    for ln in re.split(r'[.;\n]', joined_context):
        k, v = _kv_from_line(ln)
        if k and v:
            raw = _norm(ln)
            claims.append({
                "key": k,
                "value": v,
                "raw": raw,
                "claim_id": _claim_id(f"{k}::{v}")
            })
    return claims

# -------------------------------
# Utilities
# -------------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _safe_json_loads(s: str, default: Any) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return default

def _truncate_tokens(text: str, max_tokens: int) -> str:
    """
    Token-agnostic conservative truncation. If you have a tokenizer,
    swap this for token-aware truncation. Here we cap by characters (~4 chars ≈ 1 token).
    """
    approx_chars = max(256, min(len(text), max_tokens * 4))
    return text[:approx_chars]

def _pick_first_nonempty(*vals: str) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""

# -------------------------------
# Types
# -------------------------------

SearchResult = Dict[str, Any]  # {id, score, title, url?, snippet}
LoadedDoc = Dict[str, Any]     # {id, title, url?, text}
Citation = Dict[str, Any]      # {id, title, url?}

# -------------------------------
# HelperAIAgent
# -------------------------------

@dataclass
class HelperAIAgent:
    """
    A small agent providing:
      - RAG (search + load + condense) with explicit abstention if ungrounded
      - JSON-file backed memory with TTL
      - Output formatter for clean, human-readable run reports
    """
    retriever: Optional[Callable[[str, int, Optional[Dict[str, Any]]], List[SearchResult]]] = None
    loader: Optional[Callable[[str], LoadedDoc]] = None
    llm: Optional[Any] = None  # LLM model (e.g., ChatOpenAI)

    # memory store: { namespace: { key: {value: Any, updated_at: str, ttl_days: int, expires_at: Optional[str]} } }
    memory: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    # ------------- RAG -------------

    def search_corpus(self, query: str, top_k: int = 10, filters_json: str = "{}") -> Dict[str, Any]:
        """
        Semantic + keyword search over indexed corpora.
        Returns: {"results": [ {id, score, title, url?, snippet}, ... ]}
        """
        filters = _safe_json_loads(filters_json, default={})
        if self.retriever is None:
            # Observational default: if no retriever is wired, return empty (no fabrication).
            return {"results": []}

        results = self.retriever(query, top_k, filters) or []
        # Defensive normalization
        normed: List[SearchResult] = []
        for r in results:
            normed.append({
                "id": str(r.get("id", "")),
                "score": float(r.get("score", 0.0)),
                "title": str(r.get("title", "")),
                "url": r.get("url"),
                "snippet": str(r.get("snippet", "")),
            })
        return {"results": normed}

    def retrieve_context(self, document_ids: List[str], max_tokens: int = 3000) -> Dict[str, Any]:
        """
        Fetch chunked content for given doc IDs and condense into a single context window.
        Returns: {"context": str, "citations": [ {id, title, url?}, ... ]}
        """
        if not self.loader or not document_ids:
            return {"context": "", "citations": []}

        context_parts: List[str] = []
        citations: List[Citation] = []

        for doc_id in document_ids:
            try:
                doc = self.loader(doc_id)
                if not doc or not doc.get("text"):
                    continue
                text = str(doc["text"])
                # Keep per-doc a small header to maintain traceability
                header = f"[{doc.get('title','Untitled')} — {doc.get('id','')}]"
                combined = header + "\n" + text
                context_parts.append(combined)

                citations.append({
                    "id": str(doc.get("id", "")),
                    "title": str(doc.get("title", "Untitled")),
                    "url": doc.get("url"),
                })
            except Exception:
                # Strictly observational: skip bad docs, don't invent content.
                continue

        if not context_parts:
            return {"context": "", "citations": []}

        # Concatenate then truncate to budget.
        concatenated = "\n\n".join(context_parts)
        condensed = _truncate_tokens(concatenated, max_tokens=max_tokens)

        return {"context": condensed, "citations": citations}

    # -------------------------------
    # Output Formatter Helpers
    # -------------------------------

    @staticmethod
    def _strip_noise(text: str) -> str:
        """Remove HTTP logs, 'Entering chain', tool spam, and compress whitespace."""
        if not text:
            return ""
        patterns = [
            r"^> Entering new AgentExecutor chain\.\.\.\s*",
            r"^> Finished chain\.\s*",
            r"^INFO:httpx:.*$",
            r"^DEBUG:.*$",
            r"^TRACE:.*$",
            r"^WARNING:.*$",
            r"^ERROR:.*$",
            r"^Invoking:\s*`.*`.*$",          # tool call heads
            r"^Action:\s*```.*?```",          # inline JSON action blocks
        ]
        lines = text.splitlines()
        keep = []
        for ln in lines:
            if any(re.match(p, ln) for p in patterns):
                continue
            keep.append(ln)
        out = "\n".join(keep)
        # Collapse >2 blank lines
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    @staticmethod
    def _dedupe_paragraphs(text: str) -> str:
        """Remove repeated paragraphs often printed by chains."""
        if not text:
            return ""
        seen = set()
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        kept = []
        for b in blocks:
            key = re.sub(r"\s+", " ", b.lower())
            if key in seen:
                continue
            seen.add(key)
            kept.append(b)
        return "\n\n".join(kept)

    @staticmethod
    def _extract_bullets(answer: str) -> List[str]:
        bullets: List[str] = []
        for ln in (answer or "").splitlines():
            if re.match(r"^\s*[-*]\s+", ln) or re.match(r"^\s*\d+\.\s+", ln):
                bullets.append(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", ln).strip())
        return bullets

    @staticmethod
    def _mk_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
        if not rows:
            return ""
        # Markdown table
        head = "| " + " | ".join(headers) + " |"
        sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = []
        for r in rows:
            body.append("| " + " | ".join(str(r.get(h, "")).replace("\n", " ").strip()) + " |")
        return "\n".join([head, sep] + body)

    @staticmethod
    def _normalize_sources(citations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for c in citations or []:
            out.append({
                "id": str(c.get("id","")),
                "title": str(c.get("title","")).strip() or "Source",
                "url": c.get("url") or "",
            })
        # de-dupe by id or url
        seen = set()
        uniq = []
        for c in out:
            k = c["id"] or c["url"]
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)
        return uniq

    def format_run_report(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """
        Turn a noisy agent transcript into clean Markdown + JSON.
        `run` can include:
          - input (str)
          - output (str)
          - final_answer (str)
          - chat_history (str)
          - rag_log (str)
          - past_action_log (str)
          - intermediate_steps (list|str)
          - citations (list[{id,title,url?}])
        Returns: {"markdown": str, "summary_json": dict}
        """
        question = run.get("input") or ""
        raw_output = _pick_first_nonempty(run.get("output", ""), run.get("final_answer", ""))
        chat_history = run.get("chat_history", "")
        rag_log = run.get("rag_log", "")
        past_action_log = run.get("past_action_log", "")
        intermediate_steps = run.get("intermediate_steps", "")

        # Compose a raw transcript for cleanup
        transcript_parts = []
        for part in [chat_history, past_action_log, rag_log]:
            if part:
                transcript_parts.append(str(part))
        if isinstance(intermediate_steps, list):
            steps_text = "\n".join([str(s) for s in intermediate_steps])
        else:
            steps_text = str(intermediate_steps or "")
        if steps_text:
            transcript_parts.append(steps_text)

        raw_transcript = "\n".join([p for p in transcript_parts if p])
        cleaned_transcript = self._dedupe_paragraphs(self._strip_noise(raw_transcript))

        # Clean the final output and dedupe
        cleaned_output = self._dedupe_paragraphs(self._strip_noise(raw_output))

        # Extract bullets
        bullets = self._extract_bullets(cleaned_output)

        # Heuristic: build a solvent/ratio table if lines look like “X ... (a:b, v/v)”
        table_rows: List[Dict[str, str]] = []
        ratio_pat = re.compile(r"(?i)([\w\-/+ .]+?)\s*\((\d+\s*:\s*\d+(?::\d+)?)\s*,?\s*(v/v(?:/v)?|v/v/v)?\)")
        for line in cleaned_output.splitlines():
            m = ratio_pat.search(line)
            if m:
                desc = m.group(1).strip()
                ratio = m.group(2).replace(" ", "")
                units = (m.group(3) or "").strip()
                table_rows.append({"Description": desc, "Ratio": ratio, "Units": units})

        sources = self._normalize_sources(run.get("citations"))

        # Build Markdown
        md_sections: List[str] = []
        if question:
            md_sections.append(f"### Question\n{question}")
        if cleaned_output:
            md_sections.append(f"### Final Answer\n{cleaned_output}")

        if bullets:
            md_sections.append("### Key Points\n" + "\n".join([f"- {b}" for b in bullets]))

        if table_rows:
            md_sections.append("### Extracted Ratios/Table\n" + self._mk_table(table_rows, ["Description", "Ratio", "Units"]))

        if sources:
            src_lines = []
            for s in sources:
                if s["url"]:
                    src_lines.append(f"- **{s['title']}** — {s['url']}")
                else:
                    src_lines.append(f"- **{s['title']}** (ID: {s['id']})")
            md_sections.append("### Sources\n" + "\n".join(src_lines))

        if cleaned_transcript:
            md_sections.append("<details><summary>Agent Steps (raw)</summary>\n\n```\n" + cleaned_transcript + "\n```\n</details>")

        markdown = "\n\n".join(md_sections).strip()

        summary_json = {
            "question": question,
            "answer": cleaned_output,
            "key_points": bullets,
            "table": table_rows,
            "sources": sources,
            "agent_steps": cleaned_transcript,
        }
        return {"markdown": markdown, "summary_json": summary_json}

    # ------------- Answer synthesis (with auto-format) -------------

    def answer_with_rag(self, question: str, context: str, citations: Optional[List[Citation]] = None) -> Dict[str, Any]:
        """
        Generate an answer using ONLY the provided context.
        If context is empty/insufficient, abstain and explain what is needed.
        Returns:
          {
            "answer": str,
            "citations": [ ... ],
            "report_markdown": str,
            "report_json": dict,
            "knowledge_update": dict  # result of reassess_and_upsert_knowledge
          }
        """
        citations = citations or []

        if not context.strip():
            abstain = (
                "I don’t have enough grounded context to answer this yet. "
                "Please run a corpus search and provide retrieved context (or share relevant documents)."
            )
            payload = {"answer": abstain, "citations": []}
            formatted = self.format_run_report({
                "input": question, "output": abstain, "citations": []
            })
            payload["report_markdown"] = formatted["markdown"]
            payload["report_json"] = formatted["summary_json"]
            # Also run reassess (will be a no-op)
            payload["knowledge_update"] = self.reassess_and_upsert_knowledge(question, context, citations)
            return payload

        # --- LLM CALL WITH RAG ---
        if self.llm is None:
            # Fallback if no LLM is configured
            cited_titles = [c.get("title") or c.get("id") or "source" for c in citations]
            citation_note = f"Sources: {', '.join(cited_titles)}." if cited_titles else "Sources: (not provided)."
            answer = (
                f"Grounded summary (from retrieved context only):\n"
                f"{context[:1500].rstrip()}\n\n{citation_note}"
            )
        else:
            # Create prompt for the LLM
            prompt = f"""You are a scientific research assistant. Answer the following question using ONLY the information provided in the context below. 

IMPORTANT RULES:
1. Base your answer STRICTLY on the provided context
2. If the context doesn't contain enough information to answer the question, say so explicitly
3. Include specific details, methods, parameters, and values from the context
4. Cite which sources support each part of your answer by referencing the document names in square brackets

Question: {question}

Context:
{context}

Please provide a detailed, accurate answer based solely on the above context."""

            # Call the LLM
            try:
                response = self.llm.invoke(prompt)
                # Extract content from response (handles different LLM response formats)
                if hasattr(response, 'content'):
                    answer = response.content
                elif isinstance(response, str):
                    answer = response
                else:
                    answer = str(response)
                    
                # Add citation note
                cited_titles = [c.get("title") or c.get("id") or "source" for c in citations]
                if cited_titles:
                    citation_note = f"\n\nSources: {', '.join(cited_titles)}."
                    answer = answer + citation_note
                    
            except Exception as e:
                # If LLM call fails, return error message
                answer = f"Error calling LLM: {str(e)}\n\nFalling back to context summary:\n{context[:1500].rstrip()}"
        # --------------------------------

        payload = {"answer": answer, "citations": citations}

        # Auto-format: build a clean report you can render in your UI
        formatted = self.format_run_report({
            "input": question,
            "output": answer,
            "citations": citations,
            # If you have chain logs, include them here:
            # "chat_history": ..., "past_action_log": ..., "rag_log": ..., "intermediate_steps": ...
        })
        payload["report_markdown"] = formatted["markdown"]
        payload["report_json"] = formatted["summary_json"]
        # Automatically update domain knowledge
        payload["knowledge_update"] = self.reassess_and_upsert_knowledge(question, context, citations)
        return payload

    # ----------- MEMORY -----------

    def _ensure_ns(self, namespace: str) -> Dict[str, Dict[str, Any]]:
        if namespace not in self.memory:
            self.memory[namespace] = {}
        return self.memory[namespace]

    def _compute_expiry(self, ttl_days: int) -> Optional[str]:
        if ttl_days and ttl_days > 0:
            return (datetime.utcnow() + timedelta(days=int(ttl_days))).isoformat(timespec="seconds") + "Z"
        return None

    def upsert_memory(self, namespace: str, key: str, value_json: str, ttl_days: int = 0) -> Dict[str, Any]:
        """
        Create/update an entry. `value_json` must be valid JSON.
        Returns: {"status": "ok", "namespace": ..., "key": ..., "updated_at": ...}
        """
        ns = self._ensure_ns(namespace)
        value = _safe_json_loads(value_json, default=None)
        if value is None:
            return {"status": "error", "message": "value_json is not valid JSON"}

        updated_at = _now_iso()
        expires_at = self._compute_expiry(ttl_days)
        ns[key] = {
            "value": value,
            "updated_at": updated_at,
            "ttl_days": int(ttl_days),
            "expires_at": expires_at,
        }
        return {"status": "ok", "namespace": namespace, "key": key, "updated_at": updated_at}

    def get_memory(self, namespace: str, key: str) -> Dict[str, Any]:
        """
        Retrieve an entry, honoring expiry if present.
        Returns: {"found": bool, "value": obj|None, "updated_at": str|None, "ttl_days": int|None, "expires_at": str|None}
        """
        ns = self.memory.get(namespace, {})
        entry = ns.get(key)
        if not entry:
            return {"found": False, "value": None, "updated_at": None, "ttl_days": None, "expires_at": None}

        # Expiry check
        exp = entry.get("expires_at")
        if exp:
            try:
                if datetime.utcnow() > datetime.fromisoformat(exp.replace("Z", "")):
                    # expired -> delete
                    del ns[key]
                    return {"found": False, "value": None, "updated_at": None, "ttl_days": None, "expires_at": None}
            except Exception:
                # If parsing fails, treat as not expired but keep metadata intact
                pass

        return {
            "found": True,
            "value": entry.get("value"),
            "updated_at": entry.get("updated_at"),
            "ttl_days": entry.get("ttl_days"),
            "expires_at": entry.get("expires_at"),
        }

    def delete_memory(self, namespace: str, key: str) -> Dict[str, Any]:
        """
        Delete a memory entry.
        Returns: {"status": "ok", "deleted": bool}
        """
        ns = self.memory.get(namespace, {})
        existed = key in ns
        if existed:
            del ns[key]
        return {"status": "ok", "deleted": existed}

    def list_memory(self, namespace: str, prefix: str = "", limit: int = 100) -> Dict[str, Any]:
        """
        List keys within a namespace, optionally filtered by prefix.
        Returns: {"namespace": str, "keys": list[str]}
        """
        ns = self.memory.get(namespace, {})
        keys = [k for k in ns.keys() if k.startswith(prefix)]
        return {"namespace": namespace, "keys": keys[: max(1, min(limit, 1000))]}

    def load_memory_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load memory JSON from disk and merge into current store.
        File layout expected: { "namespace1": { "keyA": {...}, ... }, "namespace2": {...} }
        Returns: {"status": "ok", "namespaces_loaded": [..]}
        """
        p = Path(file_path)
        if not p.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"status": "error", "message": "Root JSON must be an object"}
            loaded = []
            for ns_name, ns_value in data.items():
                if not isinstance(ns_value, dict):
                    continue
                if ns_name not in self.memory:
                    self.memory[ns_name] = {}
                # Shallow merge of keys
                self.memory[ns_name].update(ns_value)
                loaded.append(ns_name)
            return {"status": "ok", "namespaces_loaded": loaded}
        except Exception as e:
            return {"status": "error", "message": f"Load failed: {e}"}

    def save_memory_file(self, file_path: str, namespaces: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Persist memory to disk. If `namespaces` is provided, only those are saved.
        Returns: {"status": "ok", "path": str, "namespaces_saved": [..]}
        """
        to_save: Dict[str, Any] = {}
        if namespaces:
            for ns in namespaces:
                if ns in self.memory:
                    to_save[ns] = self.memory[ns]
        else:
            to_save = self.memory

        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(to_save, indent=2, ensure_ascii=False), encoding="utf-8")
            saved = list(to_save.keys())
            return {"status": "ok", "path": str(p), "namespaces_saved": saved}
        except Exception as e:
            return {"status": 'error', "message": f"Save failed: {e}"}
        

    # === Auto-knowledge methods ===
    def _load_domain_ns(self):
        ns = "domain:knowledge"
        _ = self._ensure_ns(ns)
        return ns

    def detect_conflicts(self, ns: str, new_claims: list):
        conflicts = []
        current_keys = self.memory.get(ns, {})
        for c in new_claims:
            key = c["key"]
            val = c["value"]
            existing_blob = current_keys.get(key)
            if not existing_blob:
                continue
            try:
                existing = json.loads(existing_blob).get("value", {})
            except Exception:
                existing = {}
            saved_val = str(existing.get("value", ""))
            if saved_val and (saved_val.strip().lower() != str(val).strip().lower()):
                conflicts.append({
                    "key": key,
                    "saved_value": saved_val,
                    "new_value": val,
                    "new_claim_id": c["claim_id"]
                })
        return conflicts

    def reassess_and_upsert_knowledge(self, question: str, context: str, citations: list, ttl_days: int = 90):
        ns = self._load_domain_ns()
        claims = _collect_candidate_claims(context)
        if not claims:
            return {"status": "no_claims", "claims": []}

        conf = _confidence_from_retrieval(citations)
        prov = [{"id": c.get("id", ""), "title": c.get("title",""), "url": c.get("url","")} for c in (citations or [])]

        upserted_claims = []
        for c in claims:
            key = c["key"]
            val = c["value"]
            record = {
                "value": {
                    "value": val,
                    "claim_id": c["claim_id"],
                    "question": question,
                    "provenance": prov,
                    "confidence": conf
                }
            }
            self.upsert_memory(ns, key, json.dumps(record), ttl_days=ttl_days)
            upserted_claims.append({"key": key, "value": val, "claim_id": c["claim_id"]})

        conflicts = self.detect_conflicts(ns, claims)
        if conflicts:
            self.upsert_memory("domain:flags", f"conflicts", json.dumps(conflicts), ttl_days=ttl_days)

        return {
            "status": "ok",
            "upserted": len(claims),
            "confidence": conf,
            "conflicts": conflicts,
            "claims": upserted_claims
        }

# ... (rest of the code remains the same)
if __name__ == "__main__":
    corpus = {
        "doc1": {"id": "doc1", "title": "Lipidomics Notes", "url": None,
                 "text": "Global lipidomics overview. Solvent matrices and nanoESI guidance. Use chloroform/methanol (1:2, v/v)."},
        "doc2": {"id": "doc2", "title": "Agent Design", "url": None,
                 "text": "RAG pipeline: search -> retrieve -> answer. Memory uses JSON with TTL. Suggested ratio: methanol:chloroform (2:1, v/v)."},
    }

    def demo_retriever(query: str, top_k: int, filters: Optional[Dict[str, Any]]):
        items: List[SearchResult] = []
        for d in corpus.values():
            if any(w.lower() in d["text"].lower() for w in query.split()):
                items.append({"id": d["id"], "score": 0.9, "title": d["title"], "url": d.get("url"), "snippet": d["text"][:100]})
        return items[:top_k]

    def demo_loader(doc_id: str) -> LoadedDoc:
        return corpus.get(doc_id, {"id": doc_id, "title": "Unknown", "url": None, "text": ""})

    agent = HelperAIAgent(retriever=demo_retriever, loader=demo_loader)

    def run_helper_query(question: str, top_k: int = 8, max_tokens: int = 2000):
        sr = agent.search_corpus(question, top_k=top_k)
        ids = [r["id"] for r in sr["results"]]
        ctx = agent.retrieve_context(ids, max_tokens=max_tokens)
        rag_result = agent.answer_with_rag(question, ctx["context"], ctx["citations"])
        reassess_result = agent.reassess_and_upsert_knowledge(question, ctx["context"], ctx["citations"])
        print("=== CLEAN MARKDOWN REPORT ===")
        print(rag_result["report_markdown"])
        print("\n=== JSON SUMMARY ===")
        print(json.dumps(rag_result["report_json"], indent=2))
        print("\n=== KNOWLEDGE REASSESSMENT ===")
        print(json.dumps(reassess_result, indent=2))
        return {"rag_result": rag_result, "reassess_result": reassess_result}

    # RAG flow
    result = run_helper_query("What solvent ratios are recommended for NanoMate nanoESI?")

    # Memory flow
    agent.upsert_memory("project:lipo", "canonical_doc", json.dumps({"id": "doc1"}), ttl_days=0)
    print("\n=== MEMORY GET ===")
    print(json.dumps(agent.get_memory("project:lipo", "canonical_doc"), indent=2))
    out = agent.save_memory_file("./_demo_memory.json")
    print("\n=== MEMORY SAVE ===")
    print(json.dumps(out, indent=2))
