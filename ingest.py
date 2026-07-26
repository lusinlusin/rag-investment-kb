"""Part A · Step 1 — build the vector store from BOTH sources:
  (1) governed metric layer   data/metrics.yaml   -> type: metric    (definitions)
  (2) unstructured documents  data/docs/*.pdf      -> type: document  (narrative prose)
                                                   +  type: table     (extracted tables)

One collection holds all three, so a question can pull whichever is relevant.
PDF prose is read with pypdf; PDF *tables* are read separately with pdfplumber,
because pypdf flattens tables into unaligned number-soup the LLM can't read.
Local embeddings (Chroma's all-MiniLM-L6-v2, offline). Re-runnable via upsert.
"""
import glob
import os
import re
from pathlib import Path

import chromadb
import pdfplumber
import pypdf
import yaml

HERE = Path(__file__).parent


def load_pdf(path):
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# A data row looks like: a text label, then >=2 pure number/percent/currency cells.
_NUM = re.compile(r"^\$?-?[\d,]+\.?\d*%?$")


def _to_markdown(rows):
    """Render kept data rows (list of cell-lists) as a GitHub-flavored Markdown table.

    Real column names can't be reliably recovered from these PDFs (pdfplumber scrambles
    the header cells), so the header is generic: 'Item' for the label column, 'Value N'
    for the rest. The meaning of the value columns is carried by the prose lead, not here.
    """
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]           # pad ragged rows
    header = ["Item"] + [f"Value {i}" for i in range(1, width)]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def load_tables(path):
    """Extract each PDF table as a clean, aligned text block, tagged with its page.

    pdfplumber's "text" strategy recovers borderless financial tables that pypdf
    turns to number-soup — but it also slices surrounding prose into fake cells. So we
    keep only rows that look like real data (a text label + at least two numeric cells)
    and drop the prose noise. Returns a list of (page_number, block_text).
    """
    settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
    # Lead each table with its page's prose (a few sentences) so the block carries the
    # natural-language words a question matches on — a bare grid of numbers embeds
    # poorly, so a good caption alone isn't enough. pypdf keeps reading order, so its
    # page text starts with the section title/intro that describes the table.
    leads = []
    for page in pypdf.PdfReader(path).pages:
        prose = " ".join((page.extract_text() or "").split())
        leads.append(prose[:400])
    blocks = []
    with pdfplumber.open(path) as pdf:
        for pageno, page in enumerate(pdf.pages, 1):
            lead = leads[pageno - 1] if pageno - 1 < len(leads) else ""
            try:
                tables = page.extract_tables(settings)
            except Exception:
                continue  # skip pages pdfplumber can't grid up
            for tbl in tables:
                rows = []
                for row in tbl:
                    cells = [(c or "").replace("\n", " ").strip() for c in row]
                    cells = [c for c in cells if c]
                    numeric = sum(1 for c in cells[1:] if _NUM.match(c) or c == "N/A")
                    if len(cells) >= 2 and numeric >= 2 and any(ch.isalpha() for ch in cells[0]):
                        rows.append(cells)   # keep as a list of cells for Markdown
                if len(rows) >= 2:   # a real table has at least a couple of data rows
                    block = f"{lead}\n\nTable on page {pageno}:\n{_to_markdown(rows)}"
                    blocks.append((pageno, block))
    return blocks


def chunk(text, size=800, overlap=150):
    # Slide a fixed-size window over the text. Each step advances by
    # (size - overlap), so neighbouring chunks share `overlap` characters —
    # that shared margin keeps a sentence that straddles a cut point intact
    # in at least one chunk.
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap  # advance less than `size` -> chunks overlap
    return [c.strip() for c in out if c.strip()]


client = chromadb.PersistentClient(path=str(HERE / "chroma_db"))  # on-disk store under ./chroma_db
# No embedding_function is given, so the collection uses Chroma's default local
# model (all-MiniLM-L6-v2, 384-dim). query.py / rag.py open it the same way, so
# questions get embedded by the SAME model — required for distances to be meaningful.
col = client.get_or_create_collection("investment_kb")

docs, ids, metas = [], [], []

# (1) governed metric layer — one chunk per metric definition (never split, so a
# definition is always retrieved whole). Stable id lets re-runs upsert in place.
with open(HERE / "data" / "metrics.yaml") as f:
    metrics = yaml.safe_load(f)
for name, d in metrics.items():
    docs.append(
        f"Metric: {name}. Definition: {d['definition']} "
        f"Formula: {d['formula']}. Notes: {d.get('notes', '')}"
    )
    ids.append(f"metric::{name}")
    metas.append({"source": "governed_metric_layer", "type": "metric", "version": d.get("version", "1.0")})

# (2) unstructured documents — narrative prose (pypdf) + tables (pdfplumber)
pdf_paths = sorted(glob.glob(str(HERE / "data" / "docs" / "*.pdf")))
doc_chunks = table_chunks = 0
for path in pdf_paths:
    fname = os.path.basename(path)
    # (2a) narrative prose, chunked
    for i, ch in enumerate(chunk(load_pdf(path))):
        docs.append(ch)
        ids.append(f"{fname}::{i}")
        metas.append({"source": fname, "type": "document"})
        doc_chunks += 1
    # (2b) tables — one chunk per table, never split (a table must stay whole)
    for j, (pageno, block) in enumerate(load_tables(path)):
        docs.append(block)
        ids.append(f"{fname}::table::{j}")
        metas.append({"source": fname, "type": "table", "page": pageno})
        table_chunks += 1

# Re-run semantics: every run re-reads and re-embeds ALL documents (full overwrite,
# not incremental). Stable ids + upsert mean unchanged records are overwritten in
# place, so re-running never duplicates. Caveat: deleting a file from data/docs/ does
# NOT remove its old chunks here (upsert only adds/overwrites) — to rebuild from
# scratch, delete the chroma_db/ folder first, then run this.
#
# Upsert in batches (stays well under Chroma's per-call limit).
BATCH = 1000
for i in range(0, len(docs), BATCH):
    # Only documents are passed (no embeddings=), so Chroma embeds each one with
    # the collection's default model here, then stores text + metadata in
    # chroma.sqlite3 and the resulting vectors in the HNSW index (chroma_db/<uuid>/*.bin).
    col.upsert(documents=docs[i:i + BATCH], ids=ids[i:i + BATCH], metadatas=metas[i:i + BATCH])

print(
    f"Ingested {len(metrics)} governed metrics + {doc_chunks} document chunks "
    f"+ {table_chunks} table chunks from {len(pdf_paths)} PDF(s) "
    f"-> {col.count()} total in collection."
)
