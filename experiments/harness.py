"""Experiment harness for the RAG report.

Re-uses the project's own building blocks (ingest.load_tables / ingest.chunk, rag.SYSTEM,
rag._cite, llm.complete) so every round evaluates the real pipeline, and only swaps out
the piece a given round is about. Each round builds its own Chroma collection under
experiments/chroma_exp/ so the main ./chroma_db is never touched.

    python experiments/harness.py r1            # one round
    python experiments/harness.py r1 r2 r3 r4   # several
    python experiments/harness.py study         # retrieval-only strategy comparison (no LLM)
    python experiments/harness.py r5            # validity stress tests on the r4 pipeline
    python experiments/harness.py r6 r6v        # query rewriting + scope-strict prompt (main set, stress tests)
    python experiments/harness.py r7 r7v        # multi-query fusion (original + rewrite): final pipeline

Results land in experiments/results/<round>.json. LLM answers are cached by
(system prompt, user prompt, repeat) in experiments/results/llm_cache.jsonl, so re-running
a round re-grades without re-paying for calls.
"""
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path

import chromadb
import pypdf
import yaml

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
sys.path.insert(0, str(ROOT))

from ingest import chunk, load_tables          # noqa: E402
from rag import SYSTEM as SYSTEM_V1, _cite    # noqa: E402
from llm import complete, describe            # noqa: E402

DOCS = ROOT / "data" / "docs"
CACHE = EXP / "cache"
RESULTS = EXP / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

FUNDS = {
    "hoopp": {"file": "hoopp-2025.pdf", "name": "HOOPP (Healthcare of Ontario Pension Plan)",
              "alias": r"\bHOOPP\b|Healthcare of Ontario"},
    "aimco": {"file": "aimco-2025.pdf", "name": "AIMCo (Alberta Investment Management Corporation)",
              "alias": r"\bAIMCo\b|Alberta Investment Management"},
    "omers": {"file": "omers-2025.pdf", "name": "OMERS (Ontario Municipal Employees Retirement System)",
              "alias": r"\bOMERS\b|Ontario Municipal Employees"},
}
FILE2FUND = {v["file"]: k for k, v in FUNDS.items()}
FILE2FUND["governed_metric_layer"] = "governed"

REFUSAL = "I don't have that in the provided data."
_REFUSAL_STRICT = re.compile(r"(I\s+don['’]t|I\s+do\s+not)\s+have\s+that\s+in\s+the\s+provided\s+data", re.I)
_REFUSAL_LOOSE = re.compile(
    r"(I\s+don['’]t|I\s+do\s+not)\s+have\s+that|not\s+(available|provided|disclosed|reported|included|"
    r"present|stated|mentioned|contained)\s+in|does\s+not\s+(contain|include|provide|report|disclose|"
    r"state|mention)|doesn['’]t\s+(contain|include|provide|report|disclose|state|mention)|no\s+(information|data)"
    r"|cannot\s+be\s+determined|is\s+not\s+a\s+pension\s+plan|not\s+applicable", re.I)

# --------------------------------------------------------------------------- corpus
def _pdf_chunks(fund):
    """All chunks for one PDF (prose + tables), with page provenance, cached to JSON.

    Prose: identical windows to ingest.chunk() over the same pypdf text, but we also keep
    each window's page range (needed to score retrieval against gold pages).
    Tables: ingest.load_tables() verbatim.
    """
    out_path = CACHE / f"{fund}_chunks.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    path = DOCS / FUNDS[fund]["file"]
    pages = [(p.extract_text() or "") for p in pypdf.PdfReader(path).pages]
    text = "\n".join(pages)                      # == ingest.load_pdf(path)
    bounds, pos = [], 0                          # char offset where each page starts
    for t in pages:
        bounds.append(pos)
        pos += len(t) + 1

    def page_of(off):
        lo = 0
        for i, b in enumerate(bounds):
            if b <= off:
                lo = i
        return lo + 1

    size, overlap, start, prose = 800, 150, 0, []
    while start < len(text):
        raw = text[start:start + size]
        if raw.strip():
            prose.append({"text": raw.strip(), "page": page_of(start),
                          "page_end": page_of(min(start + size, len(text)) - 1)})
        start += size - overlap
    assert [c["text"] for c in prose] == chunk(text), "prose windows drifted from ingest.chunk()"

    chunks = [{"id": f"{FUNDS[fund]['file']}::{i}", "type": "document", **c} for i, c in enumerate(prose)]
    for j, (pageno, block) in enumerate(load_tables(str(path))):
        chunks.append({"id": f"{FUNDS[fund]['file']}::table::{j}", "type": "table", "text": block,
                       "page": pageno, "page_end": pageno})
    out_path.write_text(json.dumps(chunks))
    return chunks


def _metric_chunks():
    metrics = yaml.safe_load((ROOT / "data" / "metrics.yaml").read_text())
    return [{"id": f"metric::{n}", "type": "metric", "version": d.get("version", "1.0"),
             "text": f"Metric: {n}. Definition: {d['definition']} Formula: {d['formula']}. Notes: {d.get('notes', '')}"}
            for n, d in metrics.items()]


def corpus_stats():
    rows = {}
    for f in FUNDS:
        ch = _pdf_chunks(f)
        n_pages = len(pypdf.PdfReader(DOCS / FUNDS[f]["file"]).pages)
        rows[f] = {"pages": n_pages, "text_chunks": sum(c["type"] == "document" for c in ch),
                   "table_chunks": sum(c["type"] == "table" for c in ch),
                   "table_pages": len({c["page"] for c in ch if c["type"] == "table"})}
    return rows


_client = chromadb.PersistentClient(path=str(EXP / "chroma_exp"))
HNSW = {"ef_construction": 400, "ef_search": 400, "max_neighbors": 32}


def build_collection(name, funds, contextual=False):
    """(Re)build one experiment collection. contextual=True prepends a short header naming
    the organization, document and page to every chunk BEFORE embedding (the chunk text
    shown to the LLM gets the same header)."""
    try:
        _client.delete_collection(name)
    except Exception:
        pass
    # A larger HNSW search budget than Chroma's default (ef_search=100, M=16): with the
    # default, the contextual-header collection silently MISSED true nearest neighbours
    # (e.g. the tracking-error metric chunk) - see the ANN-recall check in the report.
    col = _client.create_collection(name, configuration={"hnsw": HNSW})
    docs, ids, metas = [], [], []
    for m in _metric_chunks():
        docs.append(m["text"]); ids.append(m["id"])
        metas.append({"source": "governed_metric_layer", "type": "metric", "version": m["version"],
                      "fund": "governed"})
    for f in funds:
        for c in _pdf_chunks(f):
            head = (f"[{FUNDS[f]['name']} - 2025 Annual Report - page {c['page']}] "
                    if contextual else "")
            docs.append(head + c["text"]); ids.append(c["id"])
            metas.append({"source": FUNDS[f]["file"], "type": c["type"], "page": c["page"],
                          "page_end": c["page_end"], "fund": f})
    for i in range(0, len(docs), 1000):
        col.upsert(documents=docs[i:i + 1000], ids=ids[i:i + 1000], metadatas=metas[i:i + 1000])
    return col


# --------------------------------------------------------------------------- retrieval
def detect_funds(q):
    return [f for f, v in FUNDS.items() if re.search(v["alias"], q, re.I)]


def _hits(res, qi=0):
    return [{"id": i, "doc": d, "meta": m, "dist": round(float(s), 4)}
            for i, d, m, s in zip(res["ids"][qi], res["documents"][qi], res["metadatas"][qi], res["distances"][qi])]


def retrieve_naive(col, q, k=6):
    return _hits(col.query(query_texts=[q], n_results=k))


_TOK = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")


class BM25:
    """Minimal Okapi BM25 over the same chunk texts the collection holds (lexical channel)."""

    def __init__(self, col, k1=1.5, b=0.75):
        got = col.get(include=["documents", "metadatas"])
        self.ids, self.docs, self.metas = got["ids"], got["documents"], got["metadatas"]
        self.by_id = {i: (d, m) for i, d, m in zip(self.ids, self.docs, self.metas)}
        self.toks = [_TOK.findall(d.lower()) for d in self.docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        self.df = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1
        self.N = len(self.toks)
        self.tf = [{} for _ in self.toks]
        for i, t in enumerate(self.toks):
            for w in t:
                self.tf[i][w] = self.tf[i].get(w, 0) + 1

    def search(self, q, n, funds=None):
        import math
        qt = set(_TOK.findall(q.lower()))
        scores = []
        for i, tf in enumerate(self.tf):
            if funds and self.metas[i].get("fund") not in funds:
                continue
            s, L = 0.0, len(self.toks[i])
            for w in qt:
                if w in tf:
                    idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * L / self.avg))
            if s > 0:
                scores.append((s, self.ids[i]))
        return [i for _, i in sorted(scores, reverse=True)[:n]]


_bm25 = {}


def _dense(col, q, n, funds=None):
    where = None if not funds else ({"fund": funds[0]} if len(funds) == 1 else {"fund": {"$in": funds}})
    return _hits(col.query(query_texts=[q], n_results=n, where=where))


def search(col, q, n, funds=None, mode="dense", pool=20, rrf_k=60):
    """dense: embedding top-n.  hybrid: reciprocal-rank fusion of dense top-`pool` and
    BM25 top-`pool` (same fund filter), keep top-n. The lexical channel rescues exact
    terms and numbers ("management expense ratio", "since inception") that a small
    sentence-embedding model blurs."""
    if mode == "dense":
        return _dense(col, q, n, funds)
    if col.name not in _bm25:
        _bm25[col.name] = BM25(col)
    bm = _bm25[col.name]
    dense = _dense(col, q, pool, funds)
    lex = bm.search(q, pool, funds)
    score = {}
    for r, h in enumerate(dense):
        score[h["id"]] = score.get(h["id"], 0) + 1 / (rrf_k + r + 1)
    for r, i in enumerate(lex):
        score[i] = score.get(i, 0) + 1 / (rrf_k + r + 1)
    dist = {h["id"]: h["dist"] for h in dense}
    top = sorted(score, key=score.get, reverse=True)[:n]
    return [{"id": i, "doc": bm.by_id[i][0], "meta": bm.by_id[i][1], "dist": dist.get(i, -1.0)} for i in top]


def retrieve_routed(col, q, k=6, per_fund=4, generic_per_fund=2, mode="dense"):
    """Fund-aware retrieval.
    1 fund named   -> filter to that fund (+ governed metrics), top-k.
    >=2 funds named -> fan out: top-`per_fund` from each named fund.
    0 funds named  -> unfiltered top-k  UNION  top-`generic_per_fund` from every fund, so a
                      generic question ("what was the net return?") sees every fund.
    """
    named = detect_funds(q)
    if len(named) == 1:
        return search(col, q, k, [named[0], "governed"], mode)
    if len(named) >= 2:
        hits = []
        for f in named:
            hits += search(col, q, per_fund, [f], mode)
        return hits
    hits = search(col, q, k, None, mode)
    seen = {h["id"] for h in hits}
    for f in FUNDS:
        try:
            for h in search(col, q, generic_per_fund, [f], mode):
                if h["id"] not in seen:
                    hits.append(h); seen.add(h["id"])
        except Exception:
            pass   # fund not in this collection
    return hits


# --------------------------------------------------------------------------- generation
SYSTEM_V2 = SYSTEM_V1 + """
- The context can mix documents from several organizations (HOOPP, AIMCo, OMERS). Each fact
  belongs ONLY to the organization named in its [source: ...] tag. Never attribute a number
  from one organization's document to another organization.
- If the question names an organization, answer only from that organization's chunks (plus
  the governed metric layer). If none of them contain the answer, reply with the refusal
  sentence above, even if another organization's chunk has a similar figure.
- If the question names no organization and the answer differs by organization, answer
  separately for each organization present in the context, labelling each one.
- If the question rests on a premise the context contradicts, say so and give the figure
  the context actually reports.
- A governed metric definition says how a metric is computed, not what its value is. Never
  compute, estimate or guess a metric value that the context does not state."""


SYSTEM_V3 = SYSTEM_V2 + """
- Answer the exact metric, scope and date asked. If the context only has a RELATED figure
  (a sub-portfolio instead of the total fund, a different date or fiscal year, a different
  asset class, an unlabeled number in a list), do not present it as the answer: reply with
  the refusal sentence, and you may add one line naming the related figure and how it
  differs. When a metric has several reported variants (e.g. Balanced Fund vs Total Fund,
  smoothed vs fair value), name the variant for every figure you give.
- Questions may be informal, abbreviated or in another language. Answer in the language of
  the question, but keep figures and [source: ...] tags exactly as in the context."""


def cite_v2(m):
    """Citation tag that also names the organization and page for prose chunks."""
    kind = m.get("type")
    if kind == "metric":
        return f"[source: governed_metric_layer, version: {m.get('version', '?')}]"
    org = m["fund"].upper() if m.get("fund") != "aimco" else "AIMCo"
    what = "table" if kind == "table" else "text"
    return f"[source: {m['source']} ({org}), {what} - page {m.get('page', '?')}]"


_cache_lock = threading.Lock()
_CACHE_FILE = RESULTS / "llm_cache.jsonl"
_cache = {}
if _CACHE_FILE.exists():
    for line in _CACHE_FILE.read_text().splitlines():
        if line.strip():
            r = json.loads(line); _cache[r["key"]] = r["answer"]


def llm(system, user, rep=0):
    key = hashlib.sha256(f"{system}\x00{user}\x00{rep}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    for attempt in range(4):
        try:
            ans = complete(system, user) or ""
            break
        except Exception as e:                     # transient API errors
            if attempt == 3:
                ans = f"<<LLM ERROR: {e}>>"
            time.sleep(2 ** attempt)
    with _cache_lock:
        _cache[key] = ans
        with open(_CACHE_FILE, "a") as fh:
            fh.write(json.dumps({"key": key, "answer": ans}) + "\n")
    return ans


def make_prompt(question, hits, cite):
    context = "\n\n".join(f"{cite(h['meta'])} {h['doc']}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"


# --------------------------------------------------------------------------- grading
def _any(alts, text):
    return any(re.search(a, text, re.I) for a in alts)


def grade(q, answer, hits):
    expect = q.get("expect", "answer")
    strict_refusal = bool(_REFUSAL_STRICT.search(answer))
    loose_refusal = bool(_REFUSAL_LOOSE.search(answer))
    groups = q.get("must", [])
    matched = [_any(g, answer) for g in groups]
    frac = sum(matched) / len(groups) if groups else 0.0
    forbid_hit = [p for p in q.get("forbid", []) if re.search(p, answer, re.I)]

    if expect == "answer":
        correct = all(matched) and not forbid_hit
    elif expect == "refuse":
        correct = loose_refusal and (not forbid_hit or strict_refusal)
    else:  # correct_premise
        correct = all(matched) and not forbid_hit
    # citations: which docs did the answer cite, and did they come from the retrieved set?
    cited = re.findall(r"\[source:\s*([^\],(]+?)\s*[,(\]]", answer)
    cited = [c.strip() for c in cited]
    cited_funds = {FILE2FUND.get(c, "unknown") for c in cited}
    retrieved_src = {h["meta"]["source"] for h in hits}
    targets = set(q.get("funds", []))
    wrong_fund_cite = bool(targets) and expect == "answer" and bool(
        {f for f in cited_funds if f not in targets | {"governed"}})
    return {
        "correct": bool(correct), "partial": round(frac, 3), "groups_matched": matched,
        "refused_strict": strict_refusal, "refused_loose": loose_refusal, "forbid_hit": forbid_hit,
        "n_citations": len(cited), "cited_funds": sorted(cited_funds),
        "invented_source": any(c not in retrieved_src for c in cited),
        "wrong_fund_citation": wrong_fund_cite,
        "false_refusal": expect == "answer" and strict_refusal and not all(matched),
    }


def retrieval_metrics(q, hits):
    targets = [f for f in q.get("funds", []) if f != "governed"]
    doc_hits = [h for h in hits if h["meta"].get("type") != "metric"]
    by_fund = {}
    for h in doc_hits:
        f = FILE2FUND.get(h["meta"]["source"])
        by_fund[f] = by_fund.get(f, 0) + 1
    out = {"n_hits": len(hits), "fund_mix": by_fund,
           "type_mix": {t: sum(h["meta"].get("type") == t for h in hits) for t in ("document", "table", "metric")},
           "metric_hit": any(h["meta"].get("type") == "metric" for h in hits)}
    if targets and doc_hits:
        out["purity"] = sum(FILE2FUND.get(h["meta"]["source"]) in targets for h in doc_hits) / len(doc_hits)
    # context sufficiency: does the retrieved text (as a whole) contain every gold fact?
    # Separates retrieval failures (insufficient context) from generation failures.
    if q.get("must") and q.get("expect", "answer") == "answer":
        blob = "\n".join(h["doc"] for h in hits)
        out["sufficient"] = all(_any(g, blob) for g in q["must"])
    gp = q.get("gold_pages") or {}
    if gp:
        cov = []
        for f, pages in gp.items():
            ok = any(FILE2FUND.get(h["meta"]["source"]) == f and
                     any(h["meta"].get("page", -1) <= p <= h["meta"].get("page_end", h["meta"].get("page", -1))
                         for p in pages) for h in doc_hits)
            cov.append(ok)
        out["gold_hit_all"] = all(cov)
        out["gold_hit_frac"] = sum(cov) / len(cov)
    return out


# --------------------------------------------------------------------------- rounds
def load_questions():
    return yaml.safe_load((EXP / "questions.yaml").read_text())


ROUNDS = {
    # name: (funds in corpus, contextual headers, retriever, k, system prompt, cite fn, question tag,
    #        repeats, LLM query rewriting before retrieval)
    "r1": (["hoopp"], False, "naive", 6, SYSTEM_V1, _cite, "r1", 2, False),
    "r2": (["hoopp", "aimco", "omers"], False, "naive", 6, SYSTEM_V1, _cite, "r2", 2, False),
    "r3": (["hoopp", "aimco", "omers"], True, "routed", 6, SYSTEM_V2, cite_v2, "r3", 2, False),
    "r4": (["hoopp", "aimco", "omers"], True, "routed_hybrid", 8, SYSTEM_V2, cite_v2, "r3", 2, False),
    "r6": (["hoopp", "aimco", "omers"], True, "routed_hybrid", 8, SYSTEM_V3, cite_v2, "r3", 2, True),
    "r7": (["hoopp", "aimco", "omers"], True, "routed_hybrid", 8, SYSTEM_V3, cite_v2, "r3", 2, "multi"),
    # held-out check: the baseline and the final pipeline on questions never used for tuning
    "ho_r2": (["hoopp", "aimco", "omers"], False, "naive", 6, SYSTEM_V1, _cite, "ho", 2, False),
    "ho_r7": (["hoopp", "aimco", "omers"], True, "routed_hybrid", 8, SYSTEM_V3, cite_v2, "ho", 2, "multi"),
}
_COLL_NAME = {(tuple(["hoopp"]), False): "exp_hoopp", (("hoopp", "aimco", "omers"), False): "exp_all",
              (("hoopp", "aimco", "omers"), True): "exp_all_ctx"}
_built = {}


def get_collection(funds, contextual):
    key = (tuple(funds), contextual)
    if key not in _built:
        name = _COLL_NAME[key]
        try:
            col = _client.get_collection(name)
            expected = len(_metric_chunks()) + sum(len(_pdf_chunks(f)) for f in funds)
            if col.count() != expected or col.configuration["hnsw"]["ef_search"] != HNSW["ef_search"]:
                raise ValueError("stale")
        except Exception:
            print(f"  building collection {name} ...", flush=True)
            col = build_collection(name, funds, contextual)
        _built[key] = col
    return _built[key]


def retrieve_multi(kind, col, q, k):
    """Multi-query retrieval: hits for the ORIGINAL question and for its LLM rewrite,
    interleaved rank by rank (dedup), so a rewrite can add evidence but never remove the
    evidence the original wording found."""
    a = retrieve(kind, col, q, k)
    rq = translate_query(q)
    if rq.strip().lower() == q.strip().lower():
        return a
    b = retrieve(kind, col, rq, k)
    cap, out, seen = max(len(a), len(b)) + 4, [], set()
    for pair in zip(a + [None] * len(b), b + [None] * len(a)):
        for h in pair:
            if h and h["id"] not in seen:
                out.append(h); seen.add(h["id"])
    return out[:cap]


def retrieve(kind, col, q, k=6):
    if kind == "routed":
        return retrieve_routed(col, q, k=k)
    if kind == "routed_hybrid":
        return retrieve_routed(col, q, k=k, per_fund=max(4, k // 2), mode="hybrid")
    if kind == "hybrid":
        return search(col, q, k, None, "hybrid")
    return retrieve_naive(col, q, k)


def retrieval_study():
    """Retrieval-only comparison (no LLM): how often does the retrieved context contain
    every gold fact (context sufficiency), per strategy, per fund / question type."""
    qs = [q for q in load_questions() if q.get("expect", "answer") == "answer" and "r2" in q["rounds"]]
    strategies = [
        ("dense k=6 (R2)", False, "naive", 6),
        ("dense k=10", False, "naive", 10),
        ("+ context headers", True, "naive", 6),
        ("+ fund routing (R3)", True, "routed", 6),
        ("hybrid BM25 k=8, no headers", False, "routed_hybrid", 8),
        ("+ hybrid BM25 k=6", True, "routed_hybrid", 6),
        ("+ hybrid BM25 k=8 (R4)", True, "routed_hybrid", 8),
        ("+ hybrid BM25 k=12", True, "routed_hybrid", 12),
    ]
    out = []
    for label, ctx, kind, k in strategies:
        col = get_collection(["hoopp", "aimco", "omers"], ctx)
        for q in qs:
            hits = retrieve(kind, col, q["question"], k)
            m = retrieval_metrics(q, hits)
            grp = q["funds"][0] if len(q["funds"]) == 1 else ("cross" if q["category"] == "cross" else q["category"])
            out.append({"strategy": label, "id": q["id"], "group": grp, "sufficient": m.get("sufficient"),
                        "purity": m.get("purity"), "n_hits": m["n_hits"],
                        "ctx_chars": sum(len(h["doc"]) for h in hits)})
        rs = [r for r in out if r["strategy"] == label]
        print(f"{label:28s} sufficiency {sum(bool(r['sufficient']) for r in rs)}/{len(rs)}  "
              f"avg chunks {sum(r['n_hits'] for r in rs)/len(rs):.1f}", flush=True)
    (RESULTS / "retrieval_study.json").write_text(json.dumps(out, indent=1))
    return out


def run_round(name, workers=8):
    funds, ctx, kind, k, system, cite, tag, reps, rewrite = ROUNDS[name]
    col = get_collection(funds, ctx)
    qs = [q for q in load_questions() if tag in q.get("rounds", [])]
    jobs = []
    for q in qs:
        if rewrite == "multi":
            hits = retrieve_multi(kind, col, q["question"], k)
        else:
            hits = retrieve(kind, col, translate_query(q["question"]) if rewrite else q["question"], k)
        prompt = make_prompt(q["question"], hits, cite)
        for rep in range(reps):
            jobs.append((q, hits, prompt, rep))
    print(f"[{name}] {len(qs)} questions x {reps} repeats = {len(jobs)} LLM calls  ({describe()})", flush=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        answers = list(ex.map(lambda j: llm(system, j[2], j[3]), jobs))
    rows = []
    for (q, hits, prompt, rep), ans in zip(jobs, answers):
        rows.append({"round": name, "id": q["id"], "rep": rep, "category": q["category"],
                     "funds": q.get("funds", []), "expect": q.get("expect", "answer"),
                     "question": q["question"], "answer": ans,
                     "retrieved": [{"id": h["id"], "fund": FILE2FUND.get(h["meta"]["source"]),
                                    "type": h["meta"].get("type"), "page": h["meta"].get("page"),
                                    "dist": h["dist"]} for h in hits],
                     "retrieval": retrieval_metrics(q, hits), "grade": grade(q, ans, hits)})
    (RESULTS / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    acc = sum(r["grade"]["correct"] for r in rows) / len(rows)
    print(f"[{name}] accuracy {acc:.1%}", flush=True)
    return rows


# --------------------------------------------------------------------------- round 4: validity
PARAPHRASES = {
    # id: [casual / abbreviated, typo-laden, Chinese]
    "H01": ["hoopp 2025 perf vs bmk?", "waht was HOOPPs net retrun in 2025 compared to benchmrak",
            "HOOPP 2025年的净回报率是多少？和基准相比如何？"],
    "H02": ["is hoopp still fully funded? smoothed ratio now vs last yr", "HOOPP funded staus end of 2025 smothed basis and prior yaer",
            "HOOPP 2025年底按平滑基础计算的融资比率是多少？上一年呢？"],
    "H09": ["hoopp usd fx exposure in cad", "HOOPP US dolar foriegn curency exposure end 2025",
            "HOOPP 2025年底的美元外汇敞口是多少（加元计）？"],
    "A01": ["aimco balanced fund 2025 return & bmk", "AIMCo balnced fund net retrun 2025 and its benchmrk",
            "AIMCo 平衡基金2025年的净回报率和基准分别是多少？"],
    "A06": ["aimco PE perf 2025 vs bmk", "AIMCo privte equity retrun 2025 and benchmark",
            "AIMCo 私募股权2025年的回报率和基准是多少？"],
    "O02": ["omers funding level eoy 2025 (smoothed / FV)", "OMERS fundd ratio end 2025 smoothed and fair valeu basis",
            "OMERS 2025年底按平滑基础和公允价值基础的融资比率分别是多少？"],
    "O05": ["omers fx exposure + hit to returns", "OMERS foriegn curency exposure 2025 and impact on retrun",
            "OMERS 2025年底外币敞口占多少？汇率变动对回报有什么影响？"],
    "X01": ["2025 returns: hoopp vs aimco vs omers", "compair 2025 net retruns HOOPP AIMCo OMERS",
            "比较 HOOPP、AIMCo 和 OMERS 2025年的净回报率。"],
}
PARA_STYLES = ["casual", "typo", "chinese"]


def _norms(text):
    """Text variants used for robust gold matching: raw, whitespace-collapsed, and with
    spaces inside numbers removed (pypdf writes '7. 2 %', '1 1.4', 'Annesley\nWallace')."""
    flat = re.sub(r"\s+", " ", text)
    tight = re.sub(r"(?<=[\d.,])\s+(?=[\d.,%])", "", flat)
    return (text, flat, tight)


def holds_gold(text, alts):
    return any(re.search(a, v, re.I) for a in alts for v in _norms(text))


def _lit_pattern(lit):
    """'7.2' -> regex that also matches '7. 2' / '7 .2' (spacing artefacts), not '17.2'."""
    body = r"\s?".join(re.escape(ch) for ch in lit)
    return re.compile(r"(?<![\d.,])" + body + r"(?![\d])")


def _gold_literal(q):
    """First must-group's first regex -> literal number string (e.g. '7\\.7\\s?%' -> '7.7')."""
    m = re.match(r"^\\?-?\\?s?\(?([\d\\.,]+)", q["must"][0][0])
    if not m:
        return None
    lit = m.group(1).replace("\\", "")
    return lit if re.fullmatch(r"[\d.,]+", lit) and any(ch.isdigit() for ch in lit) else None


def _perturb(lit):
    if "." in lit:
        dec = len(lit.split(".")[1])
        return f"{float(lit) + 1.3:.{dec}f}"
    if "," in lit:
        return f"{int(lit.replace(',', '')) + 1317:,}"
    return str(int(lit) + 13)


def run_validity(workers=8, base="r4"):
    funds, ctx, kind, k, system, cite, tag, _, translate = ROUNDS[base]
    col = get_collection(funds, ctx)
    qs = {q["id"]: q for q in load_questions()}
    rows = []

    # (a) paraphrase robustness
    jobs = []
    for qid, variants in PARAPHRASES.items():
        for style, text in zip(PARA_STYLES, variants):
            hits = (retrieve_multi(kind, col, text, k) if translate == "multi" else
                    retrieve(kind, col, translate_query(text) if translate else text, k))
            jobs.append(("paraphrase", qs[qid], style, text, hits, make_prompt(text, hits, cite)))

    # (b) context ablation: delete every retrieved chunk that holds the answer -> must refuse
    # (c) counterfactual: change the gold number inside the context -> answer must follow the context
    single = [q for q in qs.values() if q.get("expect", "answer") == "answer"
              and len(q["funds"]) == 1 and q["funds"][0] != "governed" and "r3" in q["rounds"]]
    for q in single:
        hits = (retrieve_multi(kind, col, q["question"], k) if translate == "multi" else
                retrieve(kind, col, translate_query(q["question"]) if translate else q["question"], k))
        alts = [a for g in q["must"] for a in g]
        kept = [h for h in hits if not holds_gold(h["doc"], alts)]
        jobs.append(("ablation", q, "ablation", q["question"], kept, make_prompt(q["question"], kept, cite)))
        lit = _gold_literal(q)
        pat = _lit_pattern(lit) if lit else None
        if lit and any(pat.search(h["doc"]) for h in hits):
            new = _perturb(lit)
            cf_hits = [dict(h, doc=pat.sub(new, h["doc"])) for h in hits]
            j = ("counterfactual", q, f"{lit}->{new}", q["question"], cf_hits,
                 make_prompt(q["question"], cf_hits, cite))
            jobs.append(j)

    with cf.ThreadPoolExecutor(workers) as ex:
        answers = list(ex.map(lambda j: llm(system, j[5], 0), jobs))
    for (test, q, variant, text, hits, _), ans in zip(jobs, answers):
        row = {"round": base, "test": test, "id": q["id"], "variant": variant,
               "question": text, "answer": ans,
               "retrieved": [{"id": h["id"], "fund": FILE2FUND.get(h["meta"]["source"]),
                              "type": h["meta"].get("type"), "page": h["meta"].get("page")} for h in hits]}
        if test == "paraphrase":
            row["retrieval"] = retrieval_metrics(q, hits)
            row["grade"] = grade(q, ans, hits)
        elif test == "ablation":
            alts = [a for g in q["must"] for a in g]
            row["grade"] = {"refused": bool(_REFUSAL_LOOSE.search(ans)),
                            "gold_leak": all(holds_gold(ans, g) for g in q["must"]),
                            "n_kept": len(hits)}
        else:
            lit, new = variant.split("->")
            row["grade"] = {"follows_context": bool(_lit_pattern(new).search(ans)),
                            "uses_original": bool(_lit_pattern(lit).search(ans))}
        rows.append(row)
    out = f"{base}_validity.json"
    (RESULTS / out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for t in ("paraphrase", "ablation", "counterfactual"):
        rs = [r for r in rows if r["test"] == t]
        if t == "paraphrase":
            print(f"[{out}] paraphrase accuracy {sum(r['grade']['correct'] for r in rs)}/{len(rs)}")
        elif t == "ablation":
            print(f"[{out}] ablation refusal {sum(r['grade']['refused'] for r in rs)}/{len(rs)}, "
                  f"gold leak {sum(r['grade']['gold_leak'] for r in rs)}")
        else:
            print(f"[{out}] counterfactual follows context {sum(r['grade']['follows_context'] for r in rs)}/{len(rs)}, "
                  f"uses original {sum(r['grade']['uses_original'] for r in rs)}")
    return rows


# --------------------------------------------------------------------------- round 5: query translation
TRANSLATE_SYSTEM = """Rewrite the user's question as one clear, correctly spelled English question
for searching pension-fund annual reports. Expand abbreviations (e.g. bmk -> benchmark,
PE -> private equity, FX -> foreign currency, eoy -> end of year, perf -> performance).
Keep every organization name exactly (HOOPP, AIMCo, OMERS). Output ONLY the rewritten question."""


def translate_query(q):
    return llm(TRANSLATE_SYSTEM, q, 0).strip() or q


def ann_recall(k=6):
    """Fraction of the EXACT top-k (brute-force L2 over all stored embeddings) that the
    HNSW index actually returns, averaged over every question in the bank."""
    import numpy as np
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    qs = [q["question"] for q in load_questions()]
    Q = np.array(DefaultEmbeddingFunction()(qs))
    out = {}
    for funds, ctx in [(["hoopp"], False), (["hoopp", "aimco", "omers"], False), (["hoopp", "aimco", "omers"], True)]:
        col = get_collection(funds, ctx)
        g = col.get(include=["embeddings"]); E = np.array(g["embeddings"]); ids = g["ids"]
        rec, miss = [], 0
        for qi in range(len(qs)):
            exact = [ids[i] for i in np.argsort(((E - Q[qi]) ** 2).sum(1))[:k]]
            ann = col.query(query_embeddings=[Q[qi].tolist()], n_results=k)["ids"][0]
            rec.append(len(set(exact) & set(ann)) / k); miss += exact[0] not in ann
        out[col.name] = {"recall": float(np.mean(rec)), "q_with_miss": int(sum(r < 1 for r in rec)),
                         "top1_missing": miss, "n": len(qs)}
        print(col.name, out[col.name], flush=True)
    return out


if __name__ == "__main__":
    for name in sys.argv[1:] or ["r1", "r2", "r3", "r4"]:
        if name == "r5":
            run_validity(base="r4")
        elif name in ("r6v", "r7v"):
            run_validity(base=name[:-1])
        elif name == "study":
            retrieval_study()
        elif name == "ann":
            (RESULTS / f"ann_recall_{sys.argv[-1] if len(sys.argv) > 2 else 'fixed'}.json").write_text(
                json.dumps(ann_recall(), indent=1))
        elif name == "stats":
            print(json.dumps(corpus_stats(), indent=1))
        else:
            run_round(name)
