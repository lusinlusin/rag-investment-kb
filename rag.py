"""Part A · Step 3 — retrieve governed context, then answer with the selected LLM.

Retrieval is provider-independent (Chroma, local embeddings). Generation goes
through llm.complete(), so DeepSeek / OpenAI / Claude are a one-line .env switch.
"""
from pathlib import Path

import chromadb

from llm import complete

HERE = Path(__file__).parent
_client = chromadb.PersistentClient(path=str(HERE / "chroma_db"))
_col = _client.get_or_create_collection("investment_kb")  # the collection ingest.py built

SYSTEM = """You answer questions about an investment firm's reports and metric definitions.
Rules:
- Use ONLY the provided context. If the answer is not in the context, reply exactly:
  "I don't have that in the provided data."
- When you use a metric definition, say it comes from the governed metric layer and include its version.
- Cite the source of each fact in brackets, e.g. [source: governed_metric_layer]."""


def answer(question: str, k: int = 4) -> str:
    # R (retrieve) — pull the k chunks most relevant to the question, same as query.py.
    res = _col.query(query_texts=[question], n_results=k)
    chunks, metas = res["documents"][0], res["metadatas"][0]
    # A (augment) — stitch the chunks into one context block, tagging each with its
    # source (and version, for governed metrics) so the model can cite where facts came from.
    context = "\n\n".join(
        f"[source: {m['source']}"
        + (f", version: {m['version']}" if m.get("type") == "metric" else "")
        + f"] {c}"
        for c, m in zip(chunks, metas)
    )
    # G (generate) — the model answers from this context ONLY (see SYSTEM rules above).
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return complete(SYSTEM, prompt)
