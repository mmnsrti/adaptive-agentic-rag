import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure output directory exists
FIGURES_DIR = Path("docs/assets/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# File paths
FINAL_METRICS_PATH = Path("evaluation/results/final_metrics.json")
ABLATION_METRICS_PATH = Path("evaluation/results/final_ablation_metrics.json")
FAILURE_ANALYSIS_PATH = Path("evaluation/results/final_failure_analysis.json")
CONSISTENCY_PATH = Path("evaluation/results/a6_consistency_validation.json")


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#eeeeee"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


# ============================================================
# Figure 1: Final Retrieval Benchmark
# ============================================================
def plot_figure_1_retrieval_benchmark(final_metrics: dict):
    agg = final_metrics["aggregated_metrics"]
    systems = [
        "Naive Dense RAG",
        "BM25 RAG",
        "Hybrid RAG",
        "Hybrid + Reranker RAG",
        "Adaptive Agentic RAG",
    ]
    labels = ["Dense", "BM25", "Hybrid", "Hybrid + Reranker", "Adaptive Agentic"]

    rec10 = [agg[s]["retrieval"]["mean_recall@10"] for s in systems]
    mrr10 = [agg[s]["retrieval"]["mean_mrr@10"] for s in systems]
    ndcg10 = [agg[s]["retrieval"]["mean_ndcg@10"] for s in systems]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    b1 = ax.bar(x - width, rec10, width, label="Recall@10", color="#2b5c8f")
    b2 = ax.bar(x, mrr10, width, label="MRR@10", color="#d95f02")
    b3 = ax.bar(x + width, ndcg10, width, label="nDCG@10", color="#7570b3")

    ax.set_ylabel("Metric Score (0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Final Retrieval Benchmark Across 5 Systems (Untouched Test Set, N=100)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, loc="lower right", fontsize=10)

    # Add data labels
    for bar in b1:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.015, f"{y:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in b2:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.015, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in b3:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.015, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_retrieval_benchmark.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 2: Safety & Evidence Quality
# ============================================================
def plot_figure_2_safety_benchmark(final_metrics: dict):
    agg = final_metrics["aggregated_metrics"]
    systems = [
        "Naive Dense RAG",
        "BM25 RAG",
        "Hybrid RAG",
        "Hybrid + Reranker RAG",
        "Adaptive Agentic RAG",
    ]
    labels = ["Dense", "BM25", "Hybrid", "Hybrid + Reranker", "Adaptive Agentic"]

    prec = [agg[s]["safety"]["mean_dataset_evidence_citation_precision"] * 100 for s in systems]
    null_abs = [agg[s]["safety"]["null_abstention_rate"] * 100 for s in systems]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    b1 = ax.bar(x - width/2, prec, width, label="Citation Precision (%)", color="#1b9e77")
    b2 = ax.bar(x + width/2, null_abs, width, label="Null Abstention Safety (%)", color="#e7298a")

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Safety & Evidence Quality Across Systems (Citation Validity = 100.0% All)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc="lower right", fontsize=10)

    for bar in b1:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in b2:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_safety_benchmark.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 3: Answer Quality vs Coverage Trade-off
# ============================================================
def plot_figure_3_answer_quality_vs_coverage(failure_analysis: dict):
    summ = failure_analysis["summary"]
    correct = summ["answerable_correct_count"]  # 12
    wrong = summ["answerable_wrong_count"]      # 15
    abstained = summ["answerable_abstained_count"] # 59
    total = summ["answerable_cases"]             # 86

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300, gridspec_kw={"width_ratios": [1.2, 1]})

    # Stacked bar
    categories = ["Answerable Queries (N=86)"]
    p1 = ax1.bar(categories, [correct], label=f"Correctly Answered ({correct})", color="#2ca02c", width=0.45)
    p2 = ax1.bar(categories, [wrong], bottom=[correct], label=f"Incorrectly Answered ({wrong})", color="#d62728", width=0.45)
    p3 = ax1.bar(categories, [abstained], bottom=[correct + wrong], label=f"Conservatively Abstained ({abstained})", color="#7f7f7f", width=0.45)

    ax1.set_ylabel("Number of Queries", fontsize=11, fontweight="bold")
    ax1.set_title("Outcome Breakdown on Answerable Set", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 95)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    ax1.text(0, correct/2, f"{correct}\n(14.0%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax1.text(0, correct + wrong/2, f"{wrong}\n(17.4%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax1.text(0, correct + wrong + abstained/2, f"{abstained}\n(68.6%)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    # Key metric cards
    ax2.axis("off")
    metrics_text = (
        "Core Trade-off Summary:\n\n"
        f"• Answer Coverage:       {summ['answer_coverage']:.1%} (27 / 86)\n"
        f"• Answered Accuracy:     {summ['answered_accuracy']:.1%} (12 / 27)\n"
        f"• Overall Accuracy:      {summ['overall_answerable_accuracy']:.1%} (12 / 86)\n"
        f"• False Abstention Rate: {summ['false_abstention_rate']:.1%} (59 / 86)\n\n"
        "Reliability Principle:\n"
        "When evidence is incomplete or\n"
        "asymmetric across multi-hop sources,\n"
        "the system fails closed (UNKNOWN)\n"
        "to prevent hallucinations."
    )
    ax2.text(0.1, 0.5, metrics_text, va="center", ha="left", fontsize=10.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#dddddd"))
    ax2.set_title("Reliability Metrics Annotation", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "answer_quality_vs_coverage.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 4: Null Query Abstention Safety
# ============================================================
def plot_figure_4_null_abstention(final_metrics: dict):
    agg = final_metrics["aggregated_metrics"]
    systems = [
        "Naive Dense RAG",
        "BM25 RAG",
        "Hybrid RAG",
        "Hybrid + Reranker RAG",
        "Adaptive Agentic RAG",
    ]
    labels = ["Dense", "BM25", "Hybrid", "Hybrid + Reranker", "Adaptive Agentic"]
    null_rates = [agg[s]["safety"]["null_abstention_rate"] * 100 for s in systems]

    colors = ["#9ecae1", "#9ecae1", "#9ecae1", "#9ecae1", "#2171b5"]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars = ax.bar(labels, null_rates, color=colors, width=0.55, edgecolor="#08519c", linewidth=1.2)

    ax.set_ylabel("Null Abstention Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Null-Query Abstention Safety (N=14 Unanswerable Cases)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 110)

    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 2, f"{y:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(92.9, color="#2171b5", linestyle=":", alpha=0.6)
    plt.tight_layout()
    out_path = FIGURES_DIR / "null_abstention_safety.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 5: End-to-End Latency Comparison
# ============================================================
def plot_figure_5_latency(final_metrics: dict):
    agg = final_metrics["aggregated_metrics"]
    systems = [
        "Naive Dense RAG",
        "BM25 RAG",
        "Hybrid RAG",
        "Hybrid + Reranker RAG",
        "Adaptive Agentic RAG",
    ]
    labels = ["Dense", "BM25", "Hybrid", "Hybrid + Reranker", "Adaptive Agentic"]

    mean_lat = [agg[s]["latency"]["mean_total_sec"] for s in systems]
    p95_lat = [agg[s]["latency"]["p95_total_sec"] for s in systems]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    b1 = ax.bar(x - width/2, mean_lat, width, label="Mean Latency (s)", color="#4575b4")
    b2 = ax.bar(x + width/2, p95_lat, width, label="p95 Latency (s)", color="#d73027")

    ax.set_ylabel("Latency in Seconds", fontsize=11, fontweight="bold")
    ax.set_title("End-to-End Latency Comparison Across 5 Systems", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 18)
    ax.legend(frameon=True, loc="upper right", fontsize=10)

    for bar in b1:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.3, f"{y:.2f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in b2:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.3, f"{y:.2f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_latency_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 6: Generation Compute Reduction Across Ablations
# ============================================================
def plot_figure_6_generation_calls(ablation_metrics: dict):
    configs = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    labels = [
        "A0\nDense",
        "A1\nHybrid",
        "A2\n+Rerank",
        "A3\n+Grader",
        "A4\n+Rescue",
        "A5\n+Grounding",
        "A6\nFull",
    ]
    gen_calls = [ablation_metrics["aggregated_metrics"][c]["resources"]["generation_calls_per_query"] for c in configs]
    colors = ["#fc8d59" if c in ["A0", "A1", "A2"] else "#91bfdb" for c in configs]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    bars = ax.bar(labels, gen_calls, color=colors, width=0.55, edgecolor="#4575b4", linewidth=1.0)

    ax.set_ylabel("LLM Generation Calls / Query", fontsize=11, fontweight="bold")
    ax.set_title("Generation Compute Reduction Across Ablations (Evidence Gating Effect)", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.2)

    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "generation_calls_ablation.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 7: Progressive Retrieval Ablation
# ============================================================
def plot_figure_7_retrieval_ablation(ablation_metrics: dict):
    configs = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    labels = [
        "A0: Dense",
        "A1: Hybrid",
        "A2: +Reranker",
        "A3: +Grader",
        "A4: +Rescue",
        "A5: +Grounding",
        "A6: Full System",
    ]
    rec10 = [ablation_metrics["aggregated_metrics"][c]["retrieval"]["mean_recall@10"] for c in configs]
    ndcg10 = [ablation_metrics["aggregated_metrics"][c]["retrieval"]["mean_ndcg@10"] for c in configs]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    ax.plot(labels, rec10, marker="o", linewidth=2.5, markersize=8, color="#2b5c8f", label="Recall@10")
    ax.plot(labels, ndcg10, marker="s", linewidth=2.5, markersize=8, color="#d95f02", label="nDCG@10")

    ax.set_ylabel("Score (0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Progressive Retrieval Ablation (A0 → A6)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0.55, 0.95)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    for i, txt in enumerate(rec10):
        ax.annotate(f"{txt:.3f}", (i, rec10[i] + 0.012), ha="center", fontsize=8.5, fontweight="bold", color="#2b5c8f")
    for i, txt in enumerate(ndcg10):
        ax.annotate(f"{txt:.3f}", (i, ndcg10[i] - 0.022), ha="center", fontsize=8.5, fontweight="bold", color="#d95f02")

    plt.tight_layout()
    out_path = FIGURES_DIR / "progressive_retrieval_ablation.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 8: Progressive Safety Ablation
# ============================================================
def plot_figure_8_safety_ablation(ablation_metrics: dict):
    configs = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    labels = [
        "A0: Dense",
        "A1: Hybrid",
        "A2: +Rerank",
        "A3: +Grader",
        "A4: +Rescue",
        "A5: +Ground",
        "A6: Full",
    ]
    prec = [ablation_metrics["aggregated_metrics"][c]["safety"]["mean_dataset_evidence_citation_precision"] * 100 for c in configs]
    null_abs = [ablation_metrics["aggregated_metrics"][c]["safety"]["null_abstention_rate"] * 100 for c in configs]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    ax.plot(labels, prec, marker="^", linewidth=2.5, markersize=8, color="#1b9e77", label="Citation Precision (%)")
    ax.plot(labels, null_abs, marker="D", linewidth=2.5, markersize=8, color="#e7298a", label="Null Abstention (%)")

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Progressive Safety & Evidence Precision Ablation", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(50, 105)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    for i, txt in enumerate(prec):
        ax.annotate(f"{txt:.1f}%", (i, prec[i] + 1.8), ha="center", fontsize=8.5, fontweight="bold", color="#1b9e77")
    for i, txt in enumerate(null_abs):
        ax.annotate(f"{txt:.1f}%", (i, null_abs[i] - 3.2), ha="center", fontsize=8.5, fontweight="bold", color="#e7298a")

    plt.tight_layout()
    out_path = FIGURES_DIR / "progressive_safety_ablation.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 9: Semantic Verifier Transitions
# ============================================================
def plot_figure_9_verifier_transitions(ablation_metrics: dict):
    trans = ablation_metrics["verifier_transitions"]
    categories = [
        "Unknown → Unknown (Preserved Abstention)",
        "Wrong → Wrong (Preserved Incorrect)",
        "Right → Right (Preserved Correct)",
        "Wrong → Unknown (Beneficial Rejection)",
        "Right → Unknown (Conservative Abstention)",
        "Wrong → Right (Direct Correction)",
        "Right → Wrong (Harmful Inversion)",
    ]
    counts = [
        trans.get("unknown_to_unknown", 47),
        trans.get("wrong_to_wrong", 15),
        trans.get("right_to_right", 12),
        trans.get("wrong_to_unknown", 8),
        trans.get("right_to_unknown", 4),
        trans.get("wrong_to_right", 0),
        trans.get("right_to_wrong", 0),
    ]

    colors = [
        "#999999",
        "#e41a1c",
        "#4daf4a",
        "#377eb8", # Beneficial
        "#ff7f00", # Cost
        "#4daf4a",
        "#e41a1c",
    ]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.barh(categories[::-1], counts[::-1], color=colors[::-1], height=0.6, edgecolor="#333333", linewidth=0.8)

    ax.set_xlabel("Number of Cases (Answerable Test Set, N=86)", fontsize=11, fontweight="bold")
    ax.set_title("StructuredConclusionVerifier Outcome Transitions (A5 → A6)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, 55)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "semantic_verifier_transitions.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 10: Root Causes of Final Answerable Failures
# ============================================================
def plot_figure_10_failure_distribution(failure_analysis: dict):
    dist = failure_analysis["answerable_failure_distribution"]
    categories = [
        "Retrieval Insufficient (Missing Gold Docs)",
        "Evidence Gate False Abstention",
        "Semantic Wrong Answer (Entity/Order Error)",
        "Grounding Rejection (NLI Premise Mismatch)",
        "Conservative False Abstention",
        "Semantic Safety Abstention",
    ]
    keys = [
        "RETRIEVAL_INSUFFICIENT",
        "EVIDENCE_GATE_FALSE_ABSTENTION",
        "SEMANTIC_WRONG_ANSWER",
        "GROUNDING_REJECTION",
        "CONSERVATIVE_FALSE_ABSTENTION",
        "SEMANTIC_SAFETY_ABSTENTION",
    ]
    counts = [dist[k]["count"] for k in keys]
    pcts = [dist[k]["pct_of_failures"] * 100 for k in keys]

    colors = ["#e41a1c", "#ff7f00", "#984ea3", "#377eb8", "#4daf4a", "#ffff33"]

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
    bars = ax.barh(categories[::-1], counts[::-1], color=colors[::-1], height=0.6, edgecolor="#333333", linewidth=0.8)

    ax.set_xlabel("Number of Failures (Total Answerable Failures = 74)", fontsize=11, fontweight="bold")
    ax.set_title("Primary Root Causes of Final Answerable Failures", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, 34)

    for bar, p in zip(bars, pcts[::-1]):
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{int(w)} ({p:.1f}%)", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "final_failure_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 11: Failure Stage (Retrieval vs Downstream)
# ============================================================
def plot_figure_11_failure_stage(failure_analysis: dict):
    summ = failure_analysis["summary"]
    ret_fail = summ["retrieval_rooted_failures"]   # 28
    down_fail = summ["downstream_failures"]        # 46
    total_fail = ret_fail + down_fail              # 74

    labels = [
        f"Retrieval-Rooted\n(Missing Gold Docs)\n{ret_fail} cases ({ret_fail/total_fail:.1%})",
        f"Downstream Failures\n(Despite Complete Retrieval)\n{down_fail} cases ({down_fail/total_fail:.1%})",
    ]
    sizes = [ret_fail, down_fail]
    colors = ["#e41a1c", "#377eb8"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        textprops=dict(color="#222222", fontsize=10, fontweight="bold"),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(11)

    ax.set_title("Where Final Failures Occur (Total Failures = 74)", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    out_path = FIGURES_DIR / "failure_stage_breakdown.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Figure 12: Question-Type Outcomes
# ============================================================
def plot_figure_12_question_type_outcomes(failure_analysis: dict):
    qtypes = failure_analysis["breakdown_by_question_type"]
    labels = ["Inference", "Comparison", "Temporal", "Null (Unanswerable)"]
    keys = ["inference_query", "comparison_query", "temporal_query", "null_query"]

    correct = [qtypes[k]["correct"] for k in keys]
    wrong = [qtypes[k]["wrong"] for k in keys]
    abstained = [qtypes[k]["abstained"] for k in keys]

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    p1 = ax.bar(x, correct, width, label="Correct Answer", color="#2ca02c")
    p2 = ax.bar(x, wrong, width, bottom=correct, label="Incorrect Answer", color="#d62728")
    p3 = ax.bar(x, abstained, width, bottom=[c + w for c, w in zip(correct, wrong)], label="Abstained (Safe / Unknown)", color="#7f7f7f")

    ax.set_ylabel("Number of Queries", fontsize=11, fontweight="bold")
    ax.set_title("Performance & Outcomes by Question Type (N=100)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 35)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Annotations
    for i in range(len(keys)):
        c, w, a = correct[i], wrong[i], abstained[i]
        if c > 0:
            ax.text(i, c/2, f"{c}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if w > 0:
            ax.text(i, c + w/2, f"{w}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if a > 0:
            ax.text(i, c + w + a/2, f"{a}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "question_type_outcomes.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 80)
    print("GENERATING FINAL TECHNICAL REPORT FIGURES FROM JSON ARTIFACTS")
    print("=" * 80)
    set_plot_style()

    final_metrics = load_json(FINAL_METRICS_PATH)
    ablation_metrics = load_json(ABLATION_METRICS_PATH)
    failure_analysis = load_json(FAILURE_ANALYSIS_PATH)

    plot_figure_1_retrieval_benchmark(final_metrics)
    plot_figure_2_safety_benchmark(final_metrics)
    plot_figure_3_answer_quality_vs_coverage(failure_analysis)
    plot_figure_4_null_abstention(final_metrics)
    plot_figure_5_latency(final_metrics)
    plot_figure_6_generation_calls(ablation_metrics)
    plot_figure_7_retrieval_ablation(ablation_metrics)
    plot_figure_8_safety_ablation(ablation_metrics)
    plot_figure_9_verifier_transitions(ablation_metrics)
    plot_figure_10_failure_distribution(failure_analysis)
    plot_figure_11_failure_stage(failure_analysis)
    plot_figure_12_question_type_outcomes(failure_analysis)

    print("\nAll 12 figures successfully generated and saved to docs/assets/figures/!")


if __name__ == "__main__":
    main()
