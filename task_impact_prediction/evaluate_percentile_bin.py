"""Evaluation script for year-wise citation percentile-bin prediction."""

import os
import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

import utils
from task_impact_prediction.dataset import create_evaluation_instances

NUM_BINS = 10


def assign_yearwise_bins(instances, num_bins=NUM_BINS):
    """Assign citation percentile bins within publication year.

    Returns:
      - corpus_id_to_bin: mapping corpus_id -> int in [0, num_bins-1]
      - corpus_id_to_year: mapping corpus_id -> publication year string
    """
    by_year = defaultdict(list)
    for date, instance in instances:
        year = date[:4]
        by_year[year].append((instance["corpus_id"], float(instance["gt_citations"])))

    corpus_id_to_bin = {}
    corpus_id_to_year = {}

    for year, records in by_year.items():
        citations = np.array([citation for _, citation in records], dtype=np.float32)

        if len(records) == 1:
            corpus_id, _ = records[0]
            corpus_id_to_bin[corpus_id] = 0
            corpus_id_to_year[corpus_id] = year
            continue

        order = np.argsort(citations, kind="mergesort")
        ranks = np.empty(len(citations), dtype=np.float32)
        ranks[order] = np.arange(len(citations), dtype=np.float32)

        percentiles = ranks / float(len(citations))
        bins = np.minimum((percentiles * num_bins).astype(np.int32), num_bins - 1)

        for (corpus_id, _), bin_id in zip(records, bins):
            corpus_id_to_bin[corpus_id] = int(bin_id)
            corpus_id_to_year[corpus_id] = year

    return corpus_id_to_bin, corpus_id_to_year


def evaluate_predictions(predictions, gt_bin_dict, gt_year_dict):
    """Evaluate predictions against year-wise ground-truth bins."""
    results = []

    for pred in tqdm(predictions, desc="Evaluating predictions"):
        corpus_id = pred["corpus_id"]
        if corpus_id not in gt_bin_dict:
            continue

        pred_bin = int(pred["predicted_percentile_bin"])
        gt_bin = int(gt_bin_dict[corpus_id])
        year = gt_year_dict[corpus_id]
        confidence = float(pred.get("confidence", 0.0))

        results.append({
            "corpus_id": corpus_id,
            "year": year,
            "predicted_bin": pred_bin,
            "gt_bin": gt_bin,
            "correct": bool(pred_bin == gt_bin),
            "abs_bin_error": abs(pred_bin - gt_bin),
            "confidence": confidence,
        })

    return results


def compute_aggregate_metrics(results):
    """Compute aggregate classification metrics."""
    if len(results) == 0:
        return {
            "num_instances": 0,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "mean_abs_bin_error": None,
            "mean_confidence": None,
            "confusion_matrix": None,
        }

    y_true = np.array([r["gt_bin"] for r in results], dtype=np.int32)
    y_pred = np.array([r["predicted_bin"] for r in results], dtype=np.int32)
    conf = np.array([r["confidence"] for r in results], dtype=np.float32)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_BINS))).tolist()

    return {
        "num_instances": len(results),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mean_abs_bin_error": float(np.mean(np.abs(y_pred - y_true))),
        "mean_confidence": float(np.mean(conf)),
        "confusion_matrix": cm,
    }


def compute_yearly_metrics(results):
    """Compute metrics for each publication year."""
    by_year = defaultdict(list)
    for r in results:
        by_year[r["year"]].append(r)

    yearly = {}
    for year, rows in sorted(by_year.items()):
        y_true = np.array([r["gt_bin"] for r in rows], dtype=np.int32)
        y_pred = np.array([r["predicted_bin"] for r in rows], dtype=np.int32)
        conf = np.array([r["confidence"] for r in rows], dtype=np.float32)

        yearly[year] = {
            "num_instances": len(rows),
            "accuracy": float(accuracy_score(y_true, y_pred)) if len(rows) > 0 else None,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(rows) > 0 else None,
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(rows) > 0 else None,
            "mean_abs_bin_error": float(np.mean(np.abs(y_pred - y_true))) if len(rows) > 0 else None,
            "mean_confidence": float(np.mean(conf)) if len(rows) > 0 else None,
        }

    return yearly


def main():
    parser = argparse.ArgumentParser(description="Evaluate citation percentile-bin predictions")
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID for dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--impact_months", type=int, default=12, help="Which month to evaluate")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/scored", help="Output directory for evaluation results")
    args = parser.parse_args()

    utils.log(f"Loading predictions from {args.predictions_path}")
    predictions, predictions_metadata = utils.load_json(args.predictions_path)
    utils.log(f"Loaded {len(predictions)} predictions")

    utils.log(f"Loading corpus from HuggingFace repo {args.hf_repo_id} (split={args.split})")
    all_papers, _, _ = utils.load_corpus(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embedding_type=None,
        load_sd2publications=False,
    )
    utils.log(f"Loaded {len(all_papers)} papers")

    utils.log("Creating year-wise percentile-bin ground truth")
    evaluation_instances = create_evaluation_instances(all_papers, args.impact_months)
    gt_bin_dict, gt_year_dict = assign_yearwise_bins(evaluation_instances, num_bins=NUM_BINS)
    utils.log(f"Created ground truth for {len(gt_bin_dict)} papers")

    results = evaluate_predictions(predictions, gt_bin_dict, gt_year_dict)
    aggregates = compute_aggregate_metrics(results)
    yearly_metrics = compute_yearly_metrics(results)

    utils.log("Evaluation complete:")
    utils.log(f"  Instances: {aggregates['num_instances']}")
    utils.log(f"  Accuracy: {aggregates['accuracy']:.4f}" if aggregates["accuracy"] is not None else "  Accuracy: N/A")
    utils.log(f"  Macro-F1: {aggregates['macro_f1']:.4f}" if aggregates["macro_f1"] is not None else "  Macro-F1: N/A")
    utils.log(f"  Weighted-F1: {aggregates['weighted_f1']:.4f}" if aggregates["weighted_f1"] is not None else "  Weighted-F1: N/A")
    utils.log(f"  Mean |bin error|: {aggregates['mean_abs_bin_error']:.4f}" if aggregates["mean_abs_bin_error"] is not None else "  Mean |bin error|: N/A")

    output = {
        "aggregates": aggregates,
        "per_year": yearly_metrics,
        "per_instance": results,
    }

    predictions_filename = os.path.basename(args.predictions_path)
    base, ext = os.path.splitext(predictions_filename)
    output_filename = f"{base}.eval{ext}"

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)
    utils.log(f"Saving evaluation results to {output_path}")
    utils.save_json(output, output_path, metadata=utils.update_metadata(predictions_metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
