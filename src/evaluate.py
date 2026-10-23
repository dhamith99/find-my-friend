"""
Final training + test evaluation + model registry, for one model.

    python -m src.evaluate --model resnet18       (or: dvc repro final)

Steps
  1. Read the best hyperparameters from reports/tuning/<model>/best_params.json
  2. Train with them using several seeds (params.yaml final.seeds) -> MLflow runs
     "<model>_final_s<seed>", each with the model logged
  3. Evaluate every seed on the TEST set (first and only time test is used):
     accuracy, macro-F1, per-class recall, confusion matrix, CPU latency
  4. Register ONE seed's model as  fmf-<model>  in the MLflow Model Registry and
     give it the alias @champion. The seed is chosen by VALIDATION F1, never by
     test score, so the reported test numbers stay honest.

Outputs (no faces -> kept in Git, used by the report)
  reports/evaluation/<model>/metrics.json          headline numbers (dvc metrics show)
  reports/evaluation/<model>/details.json          everything: per seed, per epoch, confusion matrix
  reports/evaluation/<model>/curves.png            loss + accuracy per epoch (all seeds)
  reports/evaluation/<model>/confusion_matrix.png  test set, registered model
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from mlflow import MlflowClient
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader

from src.data.dataset import FaceDataset
from src.models.factory import MODEL_NAMES, count_params, model_size_mb
from src.plot_style import MODEL_COLORS, MODEL_LABELS, TEXT, TRAIN_COLOR, apply_style
from src.train import load_params, run_training, setup_mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def predict(model, loader, device):
    model.eval().to(device)
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def cpu_latency_ms(model, size: int, runs: int = 50) -> float:
    """Median time for ONE image on CPU - what the Docker API will see."""
    model = model.cpu().eval()
    x = torch.randn(1, 3, size, size)
    for _ in range(5):
        model(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    return float(statistics.median(times))


def test_metrics(probs, labels, classes):
    preds = probs.argmax(1)
    recalls = recall_score(labels, preds, labels=range(len(classes)), average=None, zero_division=0)
    return {
        "test_acc": float((preds == labels).mean()),
        "test_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        **{f"test_recall_{c}": float(r) for c, r in zip(classes, recalls)},
    }


def plot_curves(histories: dict, model_name: str, path: Path):
    """Train vs validation loss and accuracy per epoch, one line per seed."""
    apply_style()
    colors = {"train": TRAIN_COLOR, "val": MODEL_COLORS[model_name]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (seed, hist) in enumerate(histories.items()):
        ep = [h["epoch"] for h in hist]
        for ax, key in zip(axes, ("loss", "acc")):
            for split, ls in (("train", "--"), ("val", "-")):
                ax.plot(ep, [h[f"{split}_{key}"] for h in hist], ls, lw=2,
                        color=colors[split], alpha=0.95 - 0.25 * i,
                        label=f"{split} (seed {seed})")
    axes[0].set_title("Loss (log scale)")
    axes[0].set_yscale("log")   # one early spike would otherwise flatten every other line
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"{MODEL_LABELS[model_name]}: training (grey, dashed) vs validation (solid)",
                 color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, classes, model_name: str, path: Path):
    apply_style(grid=False)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else TEXT)
    ax.set_title(f"{MODEL_LABELS[model_name]}: test confusion matrix", color=TEXT, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_history(run_id: str) -> list[dict]:
    """Per-epoch curves of a run, read back from MLflow."""
    client = MlflowClient()
    keys = ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1")
    series = {k: {m.step: m.value for m in client.get_metric_history(run_id, k)} for k in keys}
    epochs = sorted(series["val_loss"])
    return [{"epoch": e, **{k: series[k].get(e) for k in keys}} for e in epochs]


def register(run_id: str, model_name: str, tags: dict) -> str:
    """Register the run's model and point the @champion alias at it."""
    client = MlflowClient()
    name = f"fmf-{model_name}"
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.set_registered_model_alias(name, "champion", mv.version)
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    client.update_model_version(
        name, mv.version,
        description=f"Final {model_name}: test F1 {tags['test_f1']}, acc {tags['test_acc']}")
    return mv.version


def evaluate(model_name: str, params: dict) -> dict:
    fp = params["final"]
    tuned = json.loads(Path(f"reports/tuning/{model_name}/best_params.json").read_text())
    hp = {**tuned["best_params"], "epochs": fp["epochs"], "patience": fp["patience"]}
    if model_name == "custom_cnn":
        hp.update(pretrained=False, freeze="none")
    else:
        hp.setdefault("freeze", "none")
        hp["pretrained"] = True

    out_dir = Path("reports/evaluation") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(params["data"]["out_dir"])
    test_ds = FaceDataset(data_dir / "test.pt", train=False)
    test_dl = DataLoader(test_ds, batch_size=64)
    classes, size = test_ds.classes, int(test_ds.images.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds_out, histories, models = [], {}, {}
    for seed in fp["seeds"]:
        res = run_training(model_name, hp, params, run_name=f"{model_name}_final_s{seed}",
                           seed=seed, log_model=True)
        # load the model back from MLflow - also proves the logged model works
        model = mlflow.pytorch.load_model(f"runs:/{res['run_id']}/model", map_location="cpu")
        history = run_history(res["run_id"])
        probs, labels = predict(model, test_dl, device)
        tm = test_metrics(probs, labels, classes)
        best = history[res["best_epoch"] - 1]
        tm["generalization_gap_acc"] = best["train_acc"] - best["val_acc"]
        # add test results to the same MLflow run
        with mlflow.start_run(run_id=res["run_id"]):
            mlflow.set_tag("stage", "final")
            mlflow.log_metrics(tm)
        seeds_out.append({"seed": seed, "run_id": res["run_id"], "best_epoch": res["best_epoch"],
                          "val_f1": res["val_f1"], "val_loss": res["val_loss"],
                          "probs": probs, "labels": labels, **tm})
        histories[seed] = history
        models[seed] = model

    # choose the seed to register by VALIDATION score (ties -> lower val loss)
    chosen = max(seeds_out, key=lambda s: (s["val_f1"], -s["val_loss"]))
    cm = confusion_matrix(chosen["labels"], chosen["probs"].argmax(1), labels=range(len(classes)))
    model = models[chosen["seed"]]
    total_p, _ = count_params(model)

    def mean_std(key):
        vals = [s[key] for s in seeds_out]
        return float(np.mean(vals)), float(np.std(vals))

    acc_m, acc_s = mean_std("test_acc")
    f1_m, f1_s = mean_std("test_f1")
    summary = {
        "model": model_name,
        "hyperparameters": hp,
        "seeds": [{k: v for k, v in s.items() if k not in ("probs", "labels")} for s in seeds_out],
        "test_acc_mean": acc_m, "test_acc_std": acc_s,
        "test_f1_mean": f1_m, "test_f1_std": f1_s,
        **{f"test_recall_{c}_mean": mean_std(f"test_recall_{c}")[0] for c in classes},
        "generalization_gap_acc_mean": mean_std("generalization_gap_acc")[0],
        "latency_cpu_ms": cpu_latency_ms(model, size),
        "model_size_mb": model_size_mb(model),
        "params_total": total_p,
        "registered_seed": chosen["seed"],
        "registered_run_id": chosen["run_id"],
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "n_test": len(test_ds),
    }

    version = register(chosen["run_id"], model_name, {
        "test_f1": round(chosen["test_f1"], 4), "test_acc": round(chosen["test_acc"], 4),
        "seed": chosen["seed"], "latency_cpu_ms": round(summary["latency_cpu_ms"], 2)})
    summary["registered_model"] = f"fmf-{model_name}"
    summary["registered_version"] = int(version)

    plot_curves(histories, model_name, out_dir / "curves.png")
    plot_confusion(cm, classes, model_name, out_dir / "confusion_matrix.png")
    summary["histories"] = {str(k): v for k, v in histories.items()}
    (out_dir / "details.json").write_text(json.dumps(summary, indent=2))
    # flat headline numbers only, so `dvc metrics show` stays readable
    headline = {k: v for k, v in summary.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    (out_dir / "metrics.json").write_text(json.dumps(headline, indent=2))

    print(f"\n[{model_name}] test acc {acc_m:.3f} ± {acc_s:.3f}   test F1 {f1_m:.3f} ± {f1_s:.3f}   "
          f"CPU latency {summary['latency_cpu_ms']:.1f} ms   size {summary['model_size_mb']:.1f} MB")
    print(f"  registered fmf-{model_name} v{version} @champion (seed {chosen['seed']})")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Final training, test evaluation and registry")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    args = ap.parse_args()
    params = load_params()
    setup_mlflow(params)
    evaluate(args.model, params)


if __name__ == "__main__":
    main()
