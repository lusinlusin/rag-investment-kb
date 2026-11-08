"""Turn experiments/results/*.json into the report's figures, LaTeX tables and numbers.

    python experiments/analyze.py

Writes report/figures/*.pdf, report/tables/*.tex and report/numbers.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

EXP = Path(__file__).resolve().parent
RES = EXP / "results"
REPORT = EXP.parent / "report"
FIG = REPORT / "figures"
TAB = REPORT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Validated reference palette (dataviz skill, light mode); first three slots are
# all-pairs CVD-safe, so funds use exactly those three.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FUND_COLOR = {"hoopp": BLUE, "aimco": ORANGE, "omers": AQUA}
FUND_LABEL = {"hoopp": "HOOPP", "aimco": "AIMCo", "omers": "OMERS"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})

MAIN = ["r1", "r2", "r3", "r4", "r6", "r7"]
LABEL = {"r1": "R1\nHOOPP only", "r2": "R2\n3 reports,\nnaive", "r3": "R3\n+ fund\nrouting",
         "r4": "R4\n+ hybrid\nBM25", "r6": "R6\n+ query\nrewrite", "r7": "R7\n+ multi-\nquery"}
SHORT = {"r1": "R1", "r2": "R2", "r3": "R3", "r4": "R4", "r6": "R6", "r7": "R7"}


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def group_of(r):
    if r["category"] == "trap":
        return "trap"
    if r["category"] == "metric":
        return "metric"
    if r["category"] in ("cross", "ambiguous"):
        return "cross"
    return "single"


def pct(x):
    return f"{100 * x:.1f}"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------- per-round summary
def summarize(rows):
    s = {"n_q": len({r["id"] for r in rows}), "n": len(rows), "acc": mean(r["grade"]["correct"] for r in rows)}
    for g in ("single", "cross", "trap", "metric"):
        rs = [r for r in rows if group_of(r) == g]
        s[f"acc_{g}"] = mean(r["grade"]["correct"] for r in rs) if rs else None
    for f in ("hoopp", "aimco", "omers"):
        rs = [r for r in rows if r["funds"] == [f] and r["category"] != "trap"]
        s[f"acc_{f}"] = mean(r["grade"]["correct"] for r in rs) if rs else None
    ans = [r for r in rows if r["expect"] == "answer" and "sufficient" in r["retrieval"]]
    s["sufficiency"] = mean(bool(r["retrieval"]["sufficient"]) for r in ans)
    suff = [r for r in ans if r["retrieval"]["sufficient"]]
    insuff = [r for r in ans if not r["retrieval"]["sufficient"]]
    s["acc_given_suff"] = mean(r["grade"]["correct"] for r in suff) if suff else None
    s["acc_given_insuff"] = mean(r["grade"]["correct"] for r in insuff) if insuff else None
    s["false_refusal"] = mean(r["grade"]["refused_strict"] and not r["grade"]["correct"] for r in suff) if suff else None
    s["decomp"] = {
        "correct": mean(r["grade"]["correct"] for r in ans),
        "gen_error": mean((not r["grade"]["correct"]) and r["retrieval"]["sufficient"] for r in ans),
        "ret_error": mean((not r["grade"]["correct"]) and not r["retrieval"]["sufficient"] for r in ans),
    }
    single = [r for r in rows if len(r["funds"]) == 1 and r["funds"][0] in FUND_LABEL and "purity" in r["retrieval"]]
    s["purity"] = mean(r["retrieval"]["purity"] for r in single) if single else None
    s["wrong_fund_cite"] = mean(r["grade"]["wrong_fund_citation"] for r in rows)
    s["invented_source"] = mean(r["grade"]["invented_source"] for r in rows)
    traps = [r for r in rows if r["expect"] == "refuse"]
    s["refusal_ok"] = mean(r["grade"]["correct"] for r in traps) if traps else None
    prem = [r for r in rows if r["expect"] == "correct_premise"]
    s["premise_ok"] = mean(r["grade"]["correct"] for r in prem) if prem else None
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["id"]].append(r["grade"]["correct"])
    s["consistency"] = mean(len(set(v)) == 1 for v in by_q.values())
    s["avg_chunks"] = mean(len(r["retrieved"]) for r in rows)
    mix = defaultdict(int)
    for r in rows:
        for h in r["retrieved"]:
            mix[h["type"]] += 1
    tot = sum(mix.values())
    s["type_mix"] = {k: v / tot for k, v in mix.items()}
    s["cited_rate"] = mean(r["grade"]["n_citations"] > 0 for r in rows if not r["grade"]["refused_strict"])
    return s


def main():
    numbers = {}
    S = {n: summarize(load(n)) for n in MAIN}
    numbers["rounds"] = S
    HO = {n: summarize(load(n)) for n in ("ho_r2", "ho_r7")}
    numbers["heldout"] = HO

    # ---------------------------------------------------------------- table: main results
    def cell(v):
        return "--" if v is None else pct(v)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Metric & R1 & R2 & R3 & R4 & R6 & R7 \\", r"\midrule"]
    spec = [
        ("Questions $\\times$ repeats", lambda s: f"{s['n_q']}$\\times$2"),
        ("\\textbf{Overall accuracy (\\%)}", lambda s: "\\textbf{" + cell(s["acc"]) + "}"),
        ("\\quad Single-fund questions", lambda s: cell(s["acc_single"])),
        ("\\quad Cross-fund / ambiguous", lambda s: cell(s["acc_cross"])),
        ("\\quad Governed metrics", lambda s: cell(s["acc_metric"])),
        ("\\quad Traps (refuse / false premise)", lambda s: cell(s["acc_trap"])),
        ("Context sufficiency (\\%)", lambda s: cell(s["sufficiency"])),
        ("Accuracy $\\mid$ sufficient context", lambda s: cell(s["acc_given_suff"])),
        ("False refusal $\\mid$ sufficient context", lambda s: cell(s["false_refusal"])),
        ("Fund purity of retrieval (\\%)", lambda s: cell(s["purity"])),
        ("Wrong-fund citation (\\%)", lambda s: cell(s["wrong_fund_cite"])),
        ("Invented source tag (\\%)", lambda s: cell(s["invented_source"])),
        ("Repeat consistency (\\%)", lambda s: cell(s["consistency"])),
        ("Avg.\\ chunks in context", lambda s: f"{s['avg_chunks']:.1f}"),
    ]
    for name, fn in spec:
        lines.append(name + " & " + " & ".join(fn(S[n]) for n in MAIN) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "main_results.tex").write_text("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- table: per fund
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Source document & R1 & R2 & R3 & R4 & R6 & R7 \\",
             r"\midrule"]
    for f in ("hoopp", "aimco", "omers"):
        lines.append(FUND_LABEL[f] + " (single-fund, n=" +
                     str(len({r['id'] for r in load('r2') if r['funds'] == [f] and r['category'] != 'trap'})) +
                     ") & " + " & ".join(cell(S[n][f"acc_{f}"]) for n in MAIN) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "per_fund.tex").write_text("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- per-question matrix (appendix)
    qrows = {}
    for n in MAIN:
        for r in load(n):
            qrows.setdefault(r["id"], {"q": r["question"], "cat": r["category"]})
            qrows[r["id"]].setdefault(n, []).append(r["grade"]["correct"])
    lines = [r"\begin{longtable}{llp{7.2cm}cccccc}", r"\toprule",
             r"ID & Type & Question & R1 & R2 & R3 & R4 & R6 & R7 \\", r"\midrule", r"\endhead"]

    def mark(v):
        if v is None:
            return ""
        k = sum(v)
        return {2: r"\cmark", 1: r"\hmark", 0: r"\xmark"}[k] if len(v) == 2 else str(k)
    for qid, d in qrows.items():
        q = d["q"].replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("_", r"\_")
        lines.append(f"{qid} & {d['cat']} & {q} & " + " & ".join(mark(d.get(n)) for n in MAIN) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (TAB / "per_question.tex").write_text("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- held-out table
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Held-out set (20 questions $\times$ 2) & R2 baseline & R7 final \\",
             r"\midrule"]
    for name, key in [("Overall accuracy (\\%)", "acc"), ("\\quad Single-fund", "acc_single"),
                      ("\\quad Cross-fund / ambiguous", "acc_cross"), ("\\quad Traps", "acc_trap"),
                      ("Context sufficiency (\\%)", "sufficiency"), ("Repeat consistency (\\%)", "consistency")]:
        lines.append(f"{name} & {cell(HO['ho_r2'][key])} & {cell(HO['ho_r7'][key])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "heldout.tex").write_text("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- retrieval study
    study = load("retrieval_study")
    strategies = list(dict.fromkeys(r["strategy"] for r in study))
    st = {}
    for s_ in strategies:
        rs = [r for r in study if r["strategy"] == s_]
        st[s_] = {"suff": mean(bool(r["sufficient"]) for r in rs), "n": len(rs),
                  "chars": mean(r["ctx_chars"] for r in rs),
                  "by": {g: mean(bool(r["sufficient"]) for r in rs if r["group"] == g)
                         for g in ("hoopp", "aimco", "omers", "cross", "ambiguous")}}
    numbers["retrieval_study"] = st
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Retrieval strategy & All & HOOPP & AIMCo & OMERS & Cross & Ambig. & Context (chars) \\", r"\midrule"]
    for s_ in strategies:
        v = st[s_]
        lines.append(s_.replace("&", r"\&") + " & " + " & ".join(
            [pct(v["suff"])] + [pct(v["by"][g]) for g in ("hoopp", "aimco", "omers", "cross", "ambiguous")]) +
            f" & {v['chars']:,.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "retrieval_study.tex").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ys = list(range(len(strategies)))[::-1]
    vals = [100 * st[s_]["suff"] for s_ in strategies]
    ax.barh(ys, vals, color=BLUE, height=0.55)
    for y, v, s_ in zip(ys, vals, strategies):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", color=INK, fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(strategies)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Context sufficiency: retrieved text contains every gold fact (%)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "retrieval_study.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- fig: accuracy by round & group
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = list(range(len(MAIN)))
    series = [("Single-fund", "acc_single", BLUE), ("Cross-fund / ambiguous", "acc_cross", ORANGE),
              ("Traps", "acc_trap", AQUA), ("Overall", "acc", INK)]
    for lab, key, col in series:
        xs = [i for i, n in enumerate(MAIN) if S[n][key] is not None]
        ys_ = [100 * S[MAIN[i]][key] for i in xs]
        ax.plot(xs, ys_, color=col, lw=2 if lab != "Overall" else 2.4,
                ls="-" if lab != "Overall" else "--", marker="o", ms=5, label=lab,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in MAIN], fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.3, len(MAIN) - 0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=7.5, ncol=4)
    fig.savefig(FIG / "accuracy_by_round.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- fig: per fund (source) accuracy
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    w = 0.26
    for j, f in enumerate(("hoopp", "aimco", "omers")):
        xs, ys_ = [], []
        for i, n in enumerate(MAIN):
            v = S[n][f"acc_{f}"]
            if v is not None:
                xs.append(i + (j - 1) * w); ys_.append(100 * v)
        ax.bar(xs, ys_, width=w - 0.03, color=FUND_COLOR[f], label=FUND_LABEL[f] + " report")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in MAIN], fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy on single-fund\nquestions (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "accuracy_by_fund.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- fig: failure decomposition
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    parts = [("Correct", "correct", BLUE), ("Wrong, context was sufficient (generation)", "gen_error", ORANGE),
             ("Wrong, context insufficient (retrieval)", "ret_error", YELLOW)]
    bottom = [0] * len(MAIN)
    for lab, key, col in parts:
        vals = [100 * S[n]["decomp"][key] for n in MAIN]
        ax.bar(x, vals, bottom=bottom, color=col, width=0.55, label=lab, edgecolor="white", linewidth=1)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if key == "correct" else INK)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in MAIN], fontsize=7.5)
    ax.set_ylabel("Answerable questions (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "failure_decomposition.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- fig: retrieval source heatmap (R2)
    import numpy as np
    funds = ["hoopp", "aimco", "omers"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, n, title in zip(axes, ["r2", "r7"], ["R2 (naive dense, top-6)", "R7 (final pipeline)"]):
        M = np.zeros((3, 4))
        for r in load(n):
            if r["rep"] or len(r["funds"]) != 1 or r["funds"][0] not in funds:
                continue
            i = funds.index(r["funds"][0])
            for h in r["retrieved"]:
                j = funds.index(h["fund"]) if h["fund"] in funds else 3
                M[i, j] += 1
        M = M / M.sum(1, keepdims=True) * 100
        ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#0d366b"]),
                  vmin=0, vmax=100, aspect="auto")
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 55 else INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["HOOPP", "AIMCo", "OMERS", "metric"], fontsize=7.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels([FUND_LABEL[f] + " Qs" for f in funds] if n == "r2" else [], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    fig.text(0.5, -0.04, "Source of retrieved chunks (% of the row's retrieved chunks)", ha="center",
             fontsize=8, color=INK2)
    fig.savefig(FIG / "retrieval_source_mix.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- validity (R5 on R4, R6, R7)
    val = {}
    for base in ("r4", "r6", "r7"):
        V = load(f"{base}_validity")
        d = {}
        for style in ("casual", "typo", "chinese"):
            rs = [r for r in V if r["test"] == "paraphrase" and r["variant"] == style]
            d[style] = mean(r["grade"]["correct"] for r in rs)
        rs = [r for r in V if r["test"] == "paraphrase"]
        d["paraphrase_all"] = mean(r["grade"]["correct"] for r in rs)
        ab = [r for r in V if r["test"] == "ablation"]
        d["ablation_refusal"] = mean(r["grade"]["refused"] for r in ab)
        d["ablation_leak"] = sum(r["grade"]["gold_leak"] for r in ab)
        d["ablation_n"] = len(ab)
        cfr = [r for r in V if r["test"] == "counterfactual"]
        d["cf_follow"] = mean(r["grade"]["follows_context"] for r in cfr)
        d["cf_orig"] = sum(r["grade"]["uses_original"] for r in cfr)
        d["cf_n"] = len(cfr)
        val[base] = d
    numbers["validity"] = val
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Stress test & R5 (on R4) & R6 & R7 \\", r"\midrule"]
    for name, key in [("Paraphrase: casual / abbreviated (\\%)", "casual"), ("Paraphrase: typos (\\%)", "typo"),
                      ("Paraphrase: Chinese (\\%)", "chinese"), ("Paraphrase: all 24 (\\%)", "paraphrase_all"),
                      ("Ablation: refuses when answer removed (\\%)", "ablation_refusal"),
                      ("Counterfactual: follows edited context (\\%)", "cf_follow")]:
        lines.append(f"{name} & " + " & ".join(pct(val[b][key]) for b in ("r4", "r6", "r7")) + r" \\")
    lines.append("Ablation cases (n) & " + " & ".join(str(val[b]["ablation_n"]) for b in ("r4", "r6", "r7")) + r" \\")
    lines.append("Counterfactual cases (n) & " + " & ".join(str(val[b]["cf_n"]) for b in ("r4", "r6", "r7")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "validity.tex").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    tests = [("Casual", "casual"), ("Typos", "typo"), ("Chinese", "chinese"),
             ("Ablation:\nrefuses", "ablation_refusal"), ("Counterfactual:\nfollows context", "cf_follow")]
    cols = {"r4": BLUE, "r6": ORANGE, "r7": AQUA}
    labs = {"r4": "R5 tests on R4 pipeline", "r6": "R6 (rewrite + strict prompt)", "r7": "R7 (multi-query)"}
    w = 0.26
    for j, b in enumerate(("r4", "r6", "r7")):
        ax.bar([i + (j - 1) * w for i in range(len(tests))], [100 * val[b][k] for _, k in tests],
               width=w - 0.03, color=cols[b], label=labs[b])
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels([t for t, _ in tests], fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Pass rate (%)")
    ax.axvline(2.5, color=GRID, lw=1)
    ax.text(1, 108, "paraphrase robustness (n=8 each)", ha="center", fontsize=7.5, color=INK2)
    ax.text(3.5, 108, "groundedness", ha="center", fontsize=7.5, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "validity.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- corpus stats + ann
    numbers["corpus"] = {}
    for f in funds:
        ch = json.loads((EXP / "cache" / f"{f}_chunks.json").read_text())
        numbers["corpus"][f] = {"text": sum(c["type"] == "document" for c in ch),
                                "table": sum(c["type"] == "table" for c in ch),
                                "table_pages": len({c["page"] for c in ch if c["type"] == "table"})}
    numbers["ann_default"] = json.loads((RES / "ann_recall_default.json").read_text())
    numbers["ann_fixed"] = json.loads((RES / "ann_recall_fixed.json").read_text())
    numbers["llm_calls"] = sum(1 for _ in open(RES / "llm_cache.jsonl"))
    (REPORT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    for n in MAIN:
        s = S[n]
        print(f"{n}: acc {pct(s['acc'])} single {cell(s['acc_single'])} cross {cell(s['acc_cross'])} "
              f"trap {cell(s['acc_trap'])} suff {pct(s['sufficiency'])} acc|suff {cell(s['acc_given_suff'])} "
              f"falseref {cell(s['false_refusal'])} purity {cell(s['purity'])} wrongcite {pct(s['wrong_fund_cite'])} "
              f"invent {pct(s['invented_source'])} consist {pct(s['consistency'])} chunks {s['avg_chunks']:.1f} "
              f"H/A/O {cell(s['acc_hoopp'])}/{cell(s['acc_aimco'])}/{cell(s['acc_omers'])} mix {s['type_mix']}")
    for n, s in HO.items():
        print(n, pct(s["acc"]), cell(s["acc_single"]), cell(s["acc_cross"]), cell(s["acc_trap"]), pct(s["sufficiency"]))
    print(json.dumps(val, indent=0))


if __name__ == "__main__":
    main()
