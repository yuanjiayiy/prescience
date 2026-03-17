"""XGBoost classifier baseline for year-wise citation percentile-bin prediction (10 bins)."""

import os
import gc
import random
import argparse
import numpy as np
import torch
import xgboost as xgb
from collections import defaultdict
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import log_loss, accuracy_score, f1_score

import utils
from task_impact_prediction.dataset import (
    load_corpus_impact, create_evaluation_instances, get_papers_for_instances, build_feature_matrix
)

NUM_BINS = 10

HYPEROPT_SPACE = {
    "eta": hp.loguniform("eta", np.log(0.01), np.log(0.2)),
    "max_depth": hp.quniform("max_depth", 3, 7, 1),
    "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
    "subsample": hp.uniform("subsample", 0.5, 0.8),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.8),
    "gamma": hp.uniform("gamma", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
}


def build_model_name(args):
    """Build model name from feature flags."""
    parts = ["xgboost_percentile_bin_classifier", args.embedding_type, f"bins{NUM_BINS}"]
    if args.use_author_names:
        parts.append("author_names")
    if args.use_author_numbers:
        parts.append("author_numbers")
    if args.use_author_papers:
        parts.append("author_papers")
    if args.use_prior_work_papers:
        parts.append("prior_work_papers")
    if args.use_prior_work_numbers:
        parts.append("prior_work_numbers")
    if args.use_followup_work_paper:
        parts.append("followup_work_paper")
    return "_".join(parts)


def assign_yearwise_bins(instances, num_bins=NUM_BINS):
    """Assign citation percentile bins within publication year.

    Returns:
      - corpus_id_to_bin: mapping corpus_id -> int in [0, num_bins-1]
      - corpus_id_to_year: mapping corpus_id -> publication year (str)
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


def train_classifier(X_train, y_train, num_evals, device):
    """Train XGBoost multiclass classifier with hyperopt tuning."""
    val_size = max(1, int(len(X_train) * 0.3))
    train_size = len(X_train) - val_size
    X_tr, X_val = X_train[:train_size], X_train[train_size:]
    y_tr, y_val = y_train[:train_size], y_train[train_size:]

    def objective(params):
        trial_params = params.copy()
        trial_params["max_depth"] = int(trial_params["max_depth"])
        trial_params["min_child_weight"] = int(trial_params["min_child_weight"])

        model = xgb.XGBClassifier(
            **trial_params,
            objective="multi:softprob",
            num_class=NUM_BINS,
            eval_metric="mlogloss",
            verbosity=0,
            device=device,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)
        loss = float(log_loss(y_val, probs, labels=list(range(NUM_BINS))))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"loss": loss, "status": STATUS_OK}

    best = fmin(fn=objective, space=HYPEROPT_SPACE, algo=tpe.suggest, max_evals=num_evals, trials=Trials())
    best["max_depth"] = int(best["max_depth"])
    best["min_child_weight"] = int(best["min_child_weight"])

    utils.log(f"Best hyperparameters: {best}")

    model = xgb.XGBClassifier(
        **best,
        objective="multi:softprob",
        num_class=NUM_BINS,
        eval_metric="mlogloss",
        verbosity=0,
        device=device,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def load_or_train_model(model_path, X_train, y_train, num_evals, device):
    """Load model from disk or train a new one."""
    if os.path.exists(model_path):
        utils.log(f"Loading model from {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    utils.log(f"Training XGBoost percentile-bin classifier with {num_evals} hyperopt evaluations")
    model = train_classifier(X_train, y_train, num_evals, device)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    utils.log(f"Model saved to {model_path}")
    return model


def predict(model, X_test, test_corpus_ids, corpus_id_to_year):
    """Generate percentile-bin predictions."""
    probs = model.predict_proba(X_test)
    pred_bins = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    outputs = []
    for cid, pred_bin, confidence, prob_vec in zip(test_corpus_ids, pred_bins, confidences, probs):
        outputs.append({
            "corpus_id": cid,
            "year": corpus_id_to_year.get(cid),
            "predicted_percentile_bin": int(pred_bin),
            "confidence": float(confidence),
            "bin_probabilities": [float(x) for x in prob_vec.tolist()],
        })
    return outputs


def main():
    parser = argparse.ArgumentParser(description="XGBoost classifier for year-wise citation percentile bins")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "test"], help="Dataset split to train on")
    parser.add_argument("--train_embeddings_dir", type=str, default="data/corpus/train", help="Directory containing training split embedding files")
    parser.add_argument("--test_embeddings_dir", type=str, default="data/corpus/test", help="Directory containing test split embedding files")
    parser.add_argument("--embedding_type", type=str, default="gtr", choices=["gtr", "specter2", "grit"], help="Embedding type")
    parser.add_argument("--use_author_names", action="store_true", help="Include author name embeddings")
    parser.add_argument("--use_author_numbers", action="store_true", help="Include author h-index and citations")
    parser.add_argument("--use_author_papers", action="store_true", help="Include author prior paper embeddings")
    parser.add_argument("--use_prior_work_papers", action="store_true", help="Include key reference embeddings")
    parser.add_argument("--use_prior_work_numbers", action="store_true", help="Include key reference citation counts")
    parser.add_argument("--use_followup_work_paper", action="store_true", help="Include paper embedding")
    parser.add_argument("--impact_months", type=int, default=12, help="Number of months used for citation target")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of hyperopt evaluations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/predictions", help="Output directory")
    args = parser.parse_args()

    feature_flags = [
        args.use_author_names, args.use_author_numbers, args.use_author_papers,
        args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper
    ]
    if not any(feature_flags):
        parser.error("At least one feature flag must be set (e.g., --use_followup_work_paper)")

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = build_model_name(args)
    model_path = os.path.join(args.output_dir, "models", f"{model_name}.model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    author_embedding_cache = {}

    utils.log(f"Loading training corpus from {args.hf_repo_id} (split={args.train_split})")
    train_papers, train_dict, train_embeddings, _ = load_corpus_impact(
        hf_repo_id=args.hf_repo_id,
        split=args.train_split,
        embeddings_dir=args.train_embeddings_dir,
        embedding_type=args.embedding_type,
    )
    utils.log(f"Loaded {len(train_papers)} training papers")

    utils.log(f"Loading test corpus from {args.hf_repo_id} (split={args.split})")
    test_papers, test_dict, test_embeddings, test_metadata = load_corpus_impact(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embeddings_dir=args.test_embeddings_dir,
        embedding_type=args.embedding_type,
    )
    utils.log(f"Loaded {len(test_papers)} test papers")

    utils.log("Creating evaluation instances")
    train_instances = create_evaluation_instances(train_papers, args.impact_months)
    test_instances = create_evaluation_instances(test_papers, args.impact_months)
    utils.log(f"Train instances: {len(train_instances)}, Test instances: {len(test_instances)}")

    train_corpus_id_to_bin, _ = assign_yearwise_bins(train_instances, num_bins=NUM_BINS)
    test_corpus_id_to_bin, test_corpus_id_to_year = assign_yearwise_bins(test_instances, num_bins=NUM_BINS)

    X_train = y_train = None
    if not os.path.exists(model_path):
        train_papers_filtered = get_papers_for_instances(train_instances, train_dict, require_key_references=True)
        utils.log(f"Building training features for {len(train_papers_filtered)} papers")
        X_train, train_corpus_ids = build_feature_matrix(
            train_papers_filtered, train_dict, train_embeddings, args.embedding_type,
            args.use_author_names, args.use_author_numbers, args.use_author_papers,
            args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper,
            author_embedding_cache, desc="Training features"
        )
        y_train = np.array([train_corpus_id_to_bin[cid] for cid in train_corpus_ids], dtype=np.int32)
        utils.log(f"Training matrix shape: {X_train.shape}")

    model = load_or_train_model(model_path, X_train, y_train, args.num_evals, device)

    test_papers_list = get_papers_for_instances(test_instances, test_dict)
    utils.log(f"Building test features for {len(test_papers_list)} papers")
    X_test, test_corpus_ids = build_feature_matrix(
        test_papers_list, test_dict, test_embeddings, args.embedding_type,
        args.use_author_names, args.use_author_numbers, args.use_author_papers,
        args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper,
        author_embedding_cache, desc="Test features"
    )
    utils.log(f"Test matrix shape: {X_test.shape}")

    utils.log("Generating percentile-bin predictions")
    predictions = predict(model, X_test, test_corpus_ids, test_corpus_id_to_year)

    eval_labels = [test_corpus_id_to_bin[cid] for cid in test_corpus_ids if cid in test_corpus_id_to_bin]
    eval_preds = [p["predicted_percentile_bin"] for p in predictions if p["corpus_id"] in test_corpus_id_to_bin]
    if eval_labels and len(eval_labels) == len(eval_preds):
        acc = float(accuracy_score(eval_labels, eval_preds))
        macro_f1 = float(f1_score(eval_labels, eval_preds, average="macro"))
        utils.log(f"Bin classification accuracy: {acc:.4f}")
        utils.log(f"Bin classification macro-F1: {macro_f1:.4f}")

    output_filename = f"predictions.{model_name}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(test_metadata, args))


if __name__ == "__main__":
    main()
