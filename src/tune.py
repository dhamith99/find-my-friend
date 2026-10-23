"""
Hyperparameter tuning with Optuna, one study per model, all logged to MLflow.

    python -m src.tune --model resnet18                 (settings from params.yaml)
    python -m src.tune --model resnet18 --n-trials 5    (quick test)
    dvc repro tune                                      (all three models)

MLflow layout:   tune_<model>  (parent run: best params, plots, summary)
                   |- trial_000  (nested run: params + per-epoch curves)
                   |- trial_001 ...   pruned trials are marked "killed"

Fairness: every model gets the same number of trials, the same epoch budget,
the same seed and the same validation set. The TEST set is never used here.

Outputs (small, no faces -> kept in Git):
  reports/tuning/<model>/best_params.json
  reports/tuning/<model>/optimization_history.png
  reports/tuning/<model>/param_importances.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import mlflow
import optuna

from src.models.factory import MODEL_NAMES
from src.train import load_params, run_training, setup_mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def suggest_hparams(trial: optuna.Trial, model_name: str, tp: dict) -> dict:
    s = tp["search_space"]
    lo, hi = s["lr"][model_name]
    hp = {
        "lr": trial.suggest_float("lr", lo, hi, log=True),
        "weight_decay": trial.suggest_float("weight_decay", *s["weight_decay"], log=True),
        "dropout": trial.suggest_float("dropout", *s["dropout"]),
        "batch_size": trial.suggest_categorical("batch_size", s["batch_size"]),
        "epochs": tp["epochs"],
        "patience": tp["patience"],
    }
    if model_name == "custom_cnn":
        hp.update(pretrained=False, freeze="none")
    else:
        hp.update(pretrained=True, freeze=trial.suggest_categorical("freeze", s["freeze"]))
    return hp


def objective_score(result: dict) -> float:
    """Maximise val macro-F1. With a small val set many trials tie (e.g. 1.0),
    so a tiny val-loss penalty prefers the more confident model."""
    return result["val_f1"] - 0.01 * result["val_loss"]


def save_plots(study: optuna.Study, out_dir: Path) -> list[Path]:
    paths = []
    plots = {
        "optimization_history.png": optuna.visualization.matplotlib.plot_optimization_history,
        "param_importances.png": optuna.visualization.matplotlib.plot_param_importances,
    }
    for fname, fn in plots.items():
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
            fig.set_size_inches(8, 5)
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=120)
            plt.close(fig)
        except Exception as e:  # e.g. importances need >1 finished trial
            print(f"  (could not draw {fname}: {e})")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, f"Not available:\n{e}", ha="center", va="center", wrap=True)
            ax.axis("off")
            fig.savefig(out_dir / fname, dpi=120)   # placeholder so DVC finds the file
            plt.close(fig)
        paths.append(out_dir / fname)
    return paths


def tune(model_name: str, params: dict, n_trials: int | None = None) -> dict:
    tp = params["tune"]
    n_trials = n_trials or tp["n_trials"]
    seed = params["seed"]
    out_dir = Path("reports/tuning") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"tune_{model_name}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=tp["pruner_warmup_epochs"]),
    )

    with mlflow.start_run(run_name=f"tune_{model_name}") as parent:
        mlflow.set_tags({"model": model_name, "stage": "tuning"})
        mlflow.log_params({"model": model_name, "n_trials": n_trials, "seed": seed,
                           "epochs": tp["epochs"], "patience": tp["patience"],
                           "sampler": "TPE", "pruner": "Median"})

        def objective(trial: optuna.Trial) -> float:
            hp = suggest_hparams(trial, model_name, tp)
            result = run_training(model_name, hp, params, trial=trial,
                                  run_name=f"trial_{trial.number:03d}",
                                  nested=True, log_model=False)
            trial.set_user_attr("val_f1", result["val_f1"])
            trial.set_user_attr("val_acc", result["val_acc"])
            trial.set_user_attr("val_loss", result["val_loss"])
            trial.set_user_attr("best_epoch", result["best_epoch"])
            return objective_score(result)

        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial
        states = [t.state for t in study.trials]
        summary = {
            "model": model_name,
            "best_trial": best.number,
            "best_params": best.params,
            "best_val_f1": best.user_attrs["val_f1"],
            "best_val_acc": best.user_attrs["val_acc"],
            "best_val_loss": best.user_attrs["val_loss"],
            "best_epoch": best.user_attrs["best_epoch"],
            "n_trials": len(study.trials),
            "n_complete": states.count(optuna.trial.TrialState.COMPLETE),
            "n_pruned": states.count(optuna.trial.TrialState.PRUNED),
        }

        (out_dir / "best_params.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metrics({"best_val_f1": summary["best_val_f1"],
                            "best_val_acc": summary["best_val_acc"],
                            "best_val_loss": summary["best_val_loss"],
                            "n_complete": summary["n_complete"],
                            "n_pruned": summary["n_pruned"]})
        mlflow.log_artifact(str(out_dir / "best_params.json"))
        for p in save_plots(study, out_dir):
            mlflow.log_artifact(str(p))
        summary["mlflow_run_id"] = parent.info.run_id

    print(f"\n[{model_name}] best trial #{best.number}: val_f1={summary['best_val_f1']:.3f} "
          f"val_acc={summary['best_val_acc']:.3f}  "
          f"({summary['n_complete']} complete, {summary['n_pruned']} pruned)")
    print(f"  params: {json.dumps(best.params)}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Tune one model with Optuna")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--n-trials", type=int, help="override tune.n_trials in params.yaml")
    args = ap.parse_args()

    params = load_params()
    setup_mlflow(params)
    tune(args.model, params, args.n_trials)


if __name__ == "__main__":
    main()
