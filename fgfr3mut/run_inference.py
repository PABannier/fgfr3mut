"""Inference script to predict FGFR3 mutation."""
import argparse
from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from fgfr3mut.bootstrap import compute_ci_with_bootstrap
from fgfr3mut.chowder import Chowder
from fgfr3mut.dataset import TCGADataset
from fgfr3mut.utils import load_ckpt, sigmoid, SlideFeaturesDataset, pad_collate_fn


NUM_WORKERS = 8


def infer(
    weights_path: Path,
    model: torch.nn.Module,
    X: List[np.ndarray],
    X_slidenames: np.ndarray,
    X_ids: np.ndarray,
    device: str,
    parser_args: Namespace,
) -> pd.DataFrame:
    """Predict cohort."""
    all_preds = []

    dataset = SlideFeaturesDataset(
        features=X,
        labels=[-1.0] * len(X),
        metadata=[{"slidename": s, "patient_id": i} for s, i in zip(X_slidenames, X_ids)],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=parser_args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=pad_collate_fn,
    )

    models_paths = []
    for num_split in range(len(list(weights_path.glob("split_*")))):
        models_paths_ = list((weights_path / f"split_{num_split}").glob("*.pt"))
        assert len(models_paths_) > 0
        models_paths.extend(models_paths_)

    print("Models to ensemble:", len(models_paths))

    for batch in tqdm(dataloader):
        features, mask, _, metadata = batch
        features, mask = features.to(device), mask.to(device)

        for model_path in tqdm(models_paths, leave=False, desc="Ensembling…"):
            model.load_state_dict(load_ckpt(model_path, device))
            model.to(device)
            model.eval()

            with torch.inference_mode():
                preds = model(features, mask=mask)[0]
                preds = preds.cpu().numpy()

            for pred_slide, slidename, patient_id in zip(
                preds, metadata["slidename"], metadata["patient_id"]
            ):
                all_preds.append((slidename, patient_id, pred_slide[0], model_path))

    all_preds = pd.DataFrame(all_preds)
    all_preds.columns = ["slidename", "patient_id", "pred", "model_filename"]
    all_preds.set_index("slidename", inplace=True)

    all_preds["pred"] = all_preds["pred"].apply(sigmoid)

    return all_preds


def compute_metrics(all_preds: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Compute metrics based on predictions and ground truth.

    Parameters
    ----------
    all_preds : pd.DataFrame
        Predictions.

    labels : pd.Series
        Ground truth labels.

    Returns
    -------
    all_preds : pd.DataFrame
        Predictions.

    """
    labels = labels[~labels.index.duplicated(keep="first")]
    all_preds["label"] = labels.reindex(all_preds["patient_id"].values).values

    all_preds_tmp = all_preds.copy()
    all_preds_tmp.set_index("patient_id", inplace=True)

    mean_preds = all_preds_tmp.groupby(all_preds_tmp.index).mean(numeric_only=True)
    mean_preds = mean_preds[~mean_preds["label"].isna()]
    med_preds = all_preds_tmp.groupby(all_preds_tmp.index).median(numeric_only=True)
    med_preds = med_preds[~med_preds["label"].isna()]

    mean_metrics = compute_ci_with_bootstrap(
        y_true=mean_preds["label"].values,
        y_pred=mean_preds["pred"].values,
        y_pred_bin=None,
        n_repeats=1000,
    )

    print("\n** Metrics **")
    auc_text = (
        f"AUC={mean_metrics['med_auc']:.2f} "
        + f"[{mean_metrics['ci_lower_bound_auc']:.2f} - "
        + f"{mean_metrics['ci_upper_bound_auc']:.2f}]"
    )
    se_text = (
        f"Se={mean_metrics['med_se']:.2f} "
        + f"[{mean_metrics['ci_lower_bound_se']:.2f} - "
        + f"{mean_metrics['ci_upper_bound_se']:.2f}]"
    )
    sp_text = (
        f"Sp={mean_metrics['med_sp']:.2f} "
        + f"[{mean_metrics['ci_lower_bound_sp']:.2f} - "
        + f"{mean_metrics['ci_upper_bound_sp']:.2f}]"
    )
    ppv_text = (
        f"PPV={mean_metrics['med_ppv']:.2f} "
        + f"[{mean_metrics['ci_lower_bound_ppv']:.2f} - "
        + f"{mean_metrics['ci_upper_bound_ppv']:.2f}]"
    )
    npv_text = (
        f"NPV={mean_metrics['med_npv']:.2f} "
        + f"[{mean_metrics['ci_lower_bound_npv']:.2f} - "
        + f"{mean_metrics['ci_upper_bound_npv']:.2f}]"
    )
    logger.success(f"{auc_text}, {se_text} / {sp_text}, {ppv_text} / {npv_text}")

    return all_preds


def get_predictions_for_mpp(data_dir: Path, weights_path: Path):
    dataset = TCGADataset(data_dir)

    X, _, X_slidenames, X_ids = dataset.get_features(
        n_tiles=parser_args.n_tiles,
        num_workers=NUM_WORKERS,
        features_as="list",
    )
    labels = dataset.load_fgfr3_status(
        binarize=True,
        keep_cases=parser_args.keep_tcga_cases,
        loeffler_cases=parser_args.loeffler_tcga_cases,
    )

    common_ids = set(labels.index).intersection(X_ids)
    mask = [i in common_ids for i in X_ids]

    X = [X[i] for i in range(len(mask)) if mask[i]]
    X_slidenames = X_slidenames[mask]
    X_ids = X_ids[mask]
    labels = labels.loc[list(set(X_ids))]

    print(f"Inference on: n_slides={len(X)}, n_patients={len(set(X_ids))}")

    model = Chowder(in_features=1536, n_extreme=100)

    predictions = infer(
        weights_path=weights_path,
        model=model,
        X=X,
        X_slidenames=X_slidenames,
        X_ids=X_ids,
        device=parser_args.device,
        parser_args=parser_args,
    )

    return predictions, labels


def main(parser_args: Namespace):
    """Run inference."""
    data_dir = Path(parser_args.data_dir)
    weights_dir = Path(parser_args.weights_dir) if parser_args.weights_dir else data_dir / "models"

    print("\n Launching inference...")
    all_preds, labels = get_predictions_for_mpp(
        data_dir=data_dir,
        weights_path=weights_dir,
    )

    # Compute evaluation metrics with bootstrapping
    all_preds = compute_metrics(all_preds=all_preds, labels=labels)

    # Save
    save_folder = Path("./predictions")
    save_folder.mkdir(exist_ok=True, parents=True)
    suffix = ""
    if parser_args.loeffler_tcga_cases:
        suffix += "_loeffler_cases"
    else:
        suffix += f"_keep_{parser_args.keep_tcga_cases}_cases"

    filename = save_folder / f"preds_{suffix}.csv"

    all_preds.to_csv(filename)
    print(f"{filename} saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=False, help="Path to the directory containing the model weights. If not specified, the weights will be loaded from the `models` subdirectory in the data_dir.")
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu device")
    parser.add_argument(
        "--n_tiles",
        type=int,
        default=None,
        help="Number of tiles used for inference (all by default)",
    )
    parser.add_argument("--keep_tcga_cases", nargs="+", default=["MIBC"])
    parser.add_argument("--loeffler_tcga_cases", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)

    parser_args = parser.parse_args()
    print(vars(parser_args))

    main(parser_args=parser_args)
