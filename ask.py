"""Part A · Step 3 entry — interactive Q&A against the governed layer."""
from rag import answer
from llm import describe

if __name__ == "__main__":
    print(f"RAG ready [{describe()}]. Type a question (or 'quit').")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", ""}:  # quit/exit or a blank line ends the session
            break
        print("\n" + answer(q))  # full RAG round-trip: retrieve -> augment -> generate
