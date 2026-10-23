"""
Compare the three final models side by side (DVC "compare" stage).

    python -m src.compare

Reads reports/evaluation/<model>/details.json and reports/tuning/<model>/best_params.json
Writes
  reports/evaluation/comparison.json       headline numbers per model
  reports/evaluation/comparison.md         table ready to paste into the report
  reports/evaluation/comparison.png        test F1 | CPU latency | model size
  reports/evaluation/per_class_recall.png  test recall for each person + unknown
  reports/evaluation/tuning_effect.png     val F1 before (baseline) vs after Optuna
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

from src.models.factory import MODEL_NAMES
from src.plot_style import MODEL_COLORS, MODEL_LABELS, TEXT, TEXT_2, apply_style
from src.train import load_params

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("reports/evaluation")
ORDER = ["custom_cnn", "mobilenet_v3_small", "resnet18"]   # same order in every figure


def baseline_val_f1(params: dict) -> dict:
    """best_val_f1 of the Phase 2 '<model>_baseline' runs from MLflow (if available)."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]))
        out = {}
        for m in MODEL_NAMES:
            runs = mlflow.search_runs(experiment_names=[params["mlflow"]["experiment"]],
                                      filter_string=f"attributes.run_name = '{m}_baseline'",
                                      order_by=["attributes.start_time DESC"], max_results=1)
            if len(runs) and "metrics.best_val_f1" in runs:
                out[m] = float(runs["metrics.best_val_f1"].iloc[0])
        return out
    except Exception as e:
        print(f"  (baseline runs not found in MLflow: {e})")
        return {}


def bar_labels(ax, bars, fmt, errors=None):
    """Value label above each bar (above the error bar, if there is one)."""
    errors = errors or [0] * len(bars)
    for b, err in zip(bars, errors):
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height() + err),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9, color=TEXT)


def plot_comparison(rows, path):
    apply_style()
    names = [r["model"] for r in rows]
    labels = [MODEL_LABELS[n] for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    stds = [r["test_f1_std"] for r in rows]
    bars = ax.bar(labels, [r["test_f1_mean"] for r in rows], color=colors, width=0.6,
                  yerr=stds, capsize=4, error_kw={"ecolor": TEXT_2, "elinewidth": 1})
    bar_labels(ax, bars, lambda v: f"{v:.2f}", errors=stds)
    ax.set_ylim(0, 1.15)
    ax.set_title("Test macro-F1 (mean ± std over seeds)")

    ax = axes[1]
    bars = ax.bar(labels, [r["latency_cpu_ms"] for r in rows], color=colors, width=0.6)
    bar_labels(ax, bars, lambda v: f"{v:.1f} ms")
    ax.set_title("CPU latency per image (lower is better)")
    ax.set_ylim(0, max(r["latency_cpu_ms"] for r in rows) * 1.2)

    ax = axes[2]
    bars = ax.bar(labels, [r["model_size_mb"] for r in rows], color=colors, width=0.6)
    bar_labels(ax, bars, lambda v: f"{v:.1f} MB")
    ax.set_title("Model size (lower is better)")
    ax.set_ylim(0, max(r["model_size_mb"] for r in rows) * 1.2)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grouped(rows, groups, value_fn, title, ylabel, path, ymax=1.15, fmt="{:.2f}"):
    """Grouped bars: one group per x item, one bar per model."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(rows)
    x = np.arange(len(groups))
    for i, r in enumerate(rows):
        vals = [value_fn(r, g) for g in groups]
        bars = ax.bar(x + (i - (len(rows) - 1) / 2) * width, [v if v is not None else 0 for v in vals],
                      width * 0.92, color=MODEL_COLORS[r["model"]], label=MODEL_LABELS[r["model"]])
        for b, v in zip(bars, vals):
            if v is not None:
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=8, color=TEXT)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncols=len(rows), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    params = load_params()
    rows = []
    for m in ORDER:
        d = json.loads((OUT / m / "details.json").read_text())
        t = json.loads(Path(f"reports/tuning/{m}/best_params.json").read_text())
        d["tuned_val_f1"] = t["best_val_f1"]
        rows.append(d)
    classes = rows[0]["classes"]
    base = baseline_val_f1(params)

    comparison = {r["model"]: {
        "test_acc_mean": round(r["test_acc_mean"], 4), "test_acc_std": round(r["test_acc_std"], 4),
        "test_f1_mean": round(r["test_f1_mean"], 4), "test_f1_std": round(r["test_f1_std"], 4),
        "latency_cpu_ms": round(r["latency_cpu_ms"], 2), "model_size_mb": round(r["model_size_mb"], 2),
        "params_total": r["params_total"],
        "generalization_gap_acc": round(r["generalization_gap_acc_mean"], 4),
        "baseline_val_f1": round(base[r["model"]], 4) if r["model"] in base else None,
        "tuned_val_f1": round(r["tuned_val_f1"], 4),
        "registered": f"{r['registered_model']} v{r['registered_version']} @champion",
    } for r in rows}
    best = max(rows, key=lambda r: (round(r["test_f1_mean"], 3), -r["latency_cpu_ms"]))
    comparison["best_model"] = best["model"]
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2))

    # markdown table for the report
    lines = ["| Model | Test accuracy | Test macro-F1 | CPU latency | Size | Parameters | "
             "Val F1 baseline → tuned |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        c = comparison[r["model"]]
        b = f"{c['baseline_val_f1']:.2f}" if c["baseline_val_f1"] is not None else "n/a"
        lines.append(f"| {MODEL_LABELS[r['model']]} | {c['test_acc_mean']:.3f} ± {c['test_acc_std']:.3f} | "
                     f"{c['test_f1_mean']:.3f} ± {c['test_f1_std']:.3f} | {c['latency_cpu_ms']:.1f} ms | "
                     f"{c['model_size_mb']:.1f} MB | {c['params_total'] / 1e6:.2f} M | "
                     f"{b} → {c['tuned_val_f1']:.2f} |")
    lines.append(f"\nTest set: {rows[0]['n_test']} images, never used for training or tuning. "
                 f"Mean ± std over seeds {[s['seed'] for s in rows[0]['seeds']]}.")
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")

    plot_comparison(rows, OUT / "comparison.png")
    plot_grouped(rows, classes,
                 lambda r, c: r[f"test_recall_{c}_mean"],
                 "Test recall per class (how often each person is recognised correctly)",
                 "Recall", OUT / "per_class_recall.png")
    plot_grouped(rows, ["Baseline (Phase 2)", "Tuned (Optuna)"],
                 lambda r, g: (base.get(r["model"]) if g.startswith("Baseline") else r["tuned_val_f1"]),
                 "Validation macro-F1 before and after hyperparameter tuning",
                 "Validation macro-F1", OUT / "tuning_effect.png")

    print("\n".join(lines))
    print(f"\nBest model on test: {MODEL_LABELS[best['model']]}")


if __name__ == "__main__":
    main()
