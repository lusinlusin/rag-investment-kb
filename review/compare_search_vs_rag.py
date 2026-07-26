"""Side by side: retrieval-only (query) vs retrieval+LLM (RAG), same question.

[Retrieval-only]  hands back the most relevant raw chunks (you read them yourself).
[Retrieval+LLM]   the model reads the chunks, then gives a cited answer; refuses if not in the data.

    # Use the 3 built-in representative questions
    python review/compare_search_vs_rag.py

    # Use your own questions (quote each one)
    python review/compare_search_vs_rag.py "what is sharpe ratio" "what was the funded status?"

Retrieval+LLM needs an API key in .env; without it you still see the retrieval-only half.
"""
import sys
from pathlib import Path

# This script lives in review/; rag.py / llm.py / chroma_db are one level up, so
# add the project root to the import path before importing them.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import answer, _col          # noqa: E402
from llm import describe              # noqa: E402

K = 3                 # how many chunks to show for the retrieval-only half
PREVIEW = 200         # Only affects how many chars are PRINTED (bump up to see full text, e.g. 800).
                      # Does NOT affect retrieval or what's fed to the LLM — those always get the full chunk.

DEFAULT_QUESTIONS = [
    "What is our official definition of tracking error?",   # governed metric layer
    "What was the funded status, and as of when?",          # PDF annual-report narrative
    "What is the portfolio manager's dog's name?",          # not in the KB -> should refuse
]


def retrieval_only(question):
    """Print the [retrieval-only] half: the top-K raw chunks."""
    res = _col.query(query_texts=[question], n_results=K)
    hits = zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    for rank, (doc, meta, dist) in enumerate(hits, 1):
        preview = doc.replace("\n", " ").strip()[:PREVIEW]
        print(f"    #{rank}  distance={dist:.3f}  [{meta['source']}]")
        print(f"        {preview}{'…' if len(doc) > PREVIEW else ''}")


def search_plus_llm(question):
    """Print the [retrieval+LLM] half: the final answer (or a hint if the key is missing)."""
    try:
        print("   ", answer(question).replace("\n", "\n    "))
    except Exception as e:
        print(f"    (couldn't call the LLM: {e})")
        print("    Set RAG_PROVIDER + the matching API key in .env, then re-run to see this half.")


def main():
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    print(f"LLM = {describe()}   |   retrieval shows top {K} chunks, each truncated to {PREVIEW} chars\n")
    for q in questions:
        print("=" * 78)
        print("Q:", q)
        print("-" * 78)
        print("[RETRIEVAL-ONLY · query] hands back the most relevant raw chunks (you read them):")
        retrieval_only(q)
        print("\n[RETRIEVAL+LLM · rag] the model reads the chunks, then answers directly:")
        search_plus_llm(q)
        print()


if __name__ == "__main__":
    main()
