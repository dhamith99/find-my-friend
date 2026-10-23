"""
Train one model and log everything to MLflow.

Examples (from the project root, with the MLflow server running):
    python -m src.train --model custom_cnn
    python -m src.train --model resnet18 --epochs 10 --lr 0.0001
    python -m src.train --model mobilenet_v3_small --freeze head

Default settings come from params.yaml (train.models.<name>); command-line
flags override them. tune.py (Phase 3) reuses run_training() for Optuna.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import random
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataset import MEAN, STD, FaceDataset
from src.models.factory import MODEL_NAMES, build_model, count_params, model_size_mb


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_params(path="params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_mlflow(params: dict):
    uri = os.environ.get("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(params["mlflow"]["experiment"])


def data_version() -> str:
    """md5 of the data/ folder from dvc.lock - links each MLflow run to a DVC data version."""
    try:
        lock = yaml.safe_load(open("dvc.lock"))
        for out in lock["stages"]["prepare"]["outs"]:
            if out["path"] == "data":
                return out["md5"]
    except Exception:
        pass
    return "unknown"


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            total_loss += criterion(out, y).item() * len(y)
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "loss": total_loss / len(labels),
        "acc": float((preds == labels).mean()),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return {"loss": total_loss / n, "acc": correct / n}


def run_training(model_name: str, hp: dict, params: dict, *, trial=None,
                 run_name: str | None = None, nested: bool = False,
                 log_model: bool = True, seed: int | None = None) -> dict:
    """
    Train one model with hyperparameters `hp` inside an MLflow run.
    hp keys: lr, weight_decay, dropout, freeze, pretrained, batch_size, epochs, patience
    If `trial` (Optuna) is given, val macro-F1 is reported each epoch so bad trials are pruned.
    Returns best validation metrics + MLflow run id.
    """
    seed = params["seed"] if seed is None else seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(params["data"]["out_dir"])

    train_ds = FaceDataset(data_dir / "train.pt", train=True)
    val_ds = FaceDataset(data_dir / "val.pt", train=False)
    classes = train_ds.classes
    nw = params["train"]["num_workers"]
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,
                          num_workers=nw, generator=g)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=nw)

    model = build_model(model_name, len(classes), hp["dropout"],
                        hp.get("pretrained", True), hp.get("freeze", "none")).to(device)
    total_p, trainable_p = count_params(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    with mlflow.start_run(run_name=run_name or model_name, nested=nested) as run:
        mlflow.set_tags({"model": model_name, "device": str(device),
                         "stage": "trial" if trial is not None else "train"})
        mlflow.log_params({"model": model_name, "seed": seed, **hp,
                           "img_size": int(train_ds.images.shape[-1]),
                           "n_train": len(train_ds), "n_val": len(val_ds),
                           "data_md5": data_version(),
                           "params_total": total_p, "params_trainable": trainable_p})

        best_f1, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
        best_val = {}
        start = time.time()

        for epoch in range(1, hp["epochs"] + 1):
            tr = train_one_epoch(model, train_dl, device, criterion, optimizer)
            va = evaluate(model, val_dl, device, criterion)
            scheduler.step()

            mlflow.log_metrics({"train_loss": tr["loss"], "train_acc": tr["acc"],
                                "val_loss": va["loss"], "val_acc": va["acc"],
                                "val_f1": va["f1"], "lr": scheduler.get_last_lr()[0]},
                               step=epoch)
            print(f"[{model_name}] epoch {epoch:2d}  train_loss {tr['loss']:.3f} "
                  f"train_acc {tr['acc']:.3f}  val_loss {va['loss']:.3f} "
                  f"val_acc {va['acc']:.3f}  val_f1 {va['f1']:.3f}")

            # keep the best epoch (by val macro-F1, ties broken by lower val loss)
            if va["f1"] > best_f1 or (va["f1"] == best_f1 and va["loss"] < best_val.get("loss", 1e9)):
                best_f1, best_val, best_epoch = va["f1"], va, epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1

            if trial is not None:
                import optuna
                trial.report(va["f1"], epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    raise optuna.TrialPruned()

            if bad_epochs >= hp["patience"]:
                print(f"[{model_name}] early stop at epoch {epoch} (best epoch {best_epoch})")
                break

        model.load_state_dict(best_state)
        mlflow.log_metrics({"best_val_f1": best_val["f1"], "best_val_acc": best_val["acc"],
                            "best_val_loss": best_val["loss"], "best_epoch": best_epoch,
                            "train_time_s": time.time() - start,
                            "model_size_mb": model_size_mb(model)})

        if log_model:
            model = model.cpu().eval()
            size = int(train_ds.images.shape[-1])
            example = torch.zeros(1, 3, size, size)
            signature = infer_signature(example.numpy(), model(example).detach().numpy())
            with tempfile.TemporaryDirectory() as tmp:
                meta = Path(tmp) / "preprocess.json"
                meta.write_text(json.dumps({"classes": classes, "img_size": size,
                                            "mean": MEAN, "std": STD}, indent=2))
                mlflow.log_artifact(str(meta))
            log_kwargs = {}
            # MLflow 3.x defaults to the 'pt2' format; 'pickle' loads back as a normal nn.Module
            if "serialization_format" in inspect.signature(mlflow.pytorch.log_model).parameters:
                log_kwargs["serialization_format"] = "pickle"
            # code_paths bundles src/ with the model, so CustomCNN can be loaded anywhere (Docker)
            mlflow.pytorch.log_model(model, name="model", signature=signature,
                                     code_paths=["src"], **log_kwargs)

        return {"run_id": run.info.run_id, "best_epoch": best_epoch,
                **{f"val_{k}": v for k, v in best_val.items()}}


def main():
    params = load_params()
    ap = argparse.ArgumentParser(description="Train one Find My Friend model")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--freeze", choices=["none", "head"])
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init instead of ImageNet weights")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-name")
    args = ap.parse_args()

    tp = params["train"]
    hp = {"epochs": tp["epochs"], "batch_size": tp["batch_size"],
          "patience": tp["patience"], **tp["models"][args.model]}
    for key in ("epochs", "batch_size", "lr", "weight_decay", "dropout", "freeze"):
        val = getattr(args, key)
        if val is not None:
            hp[key] = val
    if args.no_pretrained:
        hp["pretrained"] = False

    setup_mlflow(params)
    result = run_training(args.model, hp, params, run_name=args.run_name, seed=args.seed)
    print("\nBest validation:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
