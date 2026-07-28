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
- Answer only what is asked; don't volunteer extra definitions or background.
- Cite each fact by copying, VERBATIM, the [source: ...] tag on the chunk it came from.
  Never invent a source. Call something a governed metric definition ONLY if its tag
  literally reads [source: governed_metric_layer, version: ...], and then quote that version.
- If the answer covers several rows or items, list every one as a plain bulleted list
  ("- Name: value", one per line) — don't collapse to a total. Do NOT use Markdown
  tables; the answer is read in a plain terminal where they render as unreadable pipes."""


def _cite(m):
    """Build the [source: ...] tag for a chunk, distinguishing the three kinds so the
    answer's citations reveal whether a fact came from the PDF prose (text), an extracted
    table (+ page), or the governed metric layer (+ version)."""
    src = m["source"]
    kind = m.get("type")
    if kind == "metric":
        return f"[source: {src}, version: {m.get('version', '?')}]"
    if kind == "table":
        return f"[source: {src}, table - page {m.get('page', '?')}]"
    return f"[source: {src}, text]"   # document = PDF narrative prose


def answer(question: str, k: int = 6) -> str:
    # R (retrieve) — pull the k chunks most relevant to the question. k=6 (not 3-4) gives
    # a big single table chunk room to make the cut against many competing prose fragments.
    res = _col.query(query_texts=[question], n_results=k)
    chunks, metas = res["documents"][0], res["metadatas"][0]
    # A (augment) — stitch the chunks into one context block, tagging each with a source
    # label that distinguishes prose / table / metric (see _cite) so the model's citations
    # reveal which kind each fact came from.
    context = "\n\n".join(f"{_cite(m)} {c}" for c, m in zip(chunks, metas))
    # G (generate) — the model answers from this context ONLY (see SYSTEM rules above).
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return complete(SYSTEM, prompt)
