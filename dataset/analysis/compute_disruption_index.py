"""Compute disruption index for target papers from the PreScience dataset.

Disruption index for focal paper p is computed as:
    D = (N_i - N_j) / (N_i + N_j + N_k)
where:
  - N_i: papers citing p but none of p's references
  - N_j: papers citing both p and at least one reference of p
  - N_k: papers citing at least one reference of p but not p

This script loads target papers from HuggingFace (e.g., allenai/prescience),
uses Semantic Scholar API to fetch citation lists for focal papers + references,
and writes per-paper disruption scores.
"""

import os
import argparse
import datetime
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

import utils


def parse_date(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None


def load_citation_lists(corpus_ids: List[str], citation_cache: Dict[str, List[Tuple[str, str]]], batch_size: int) -> None:
    """Fetch citations for missing corpus IDs and cache as (citer_corpus_id, citer_pub_date)."""
    missing = [cid for cid in corpus_ids if cid not in citation_cache]
    if not missing:
        return

    records = utils.s2_batch_lookup(
        [f"CorpusId:{cid}" for cid in missing],
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "citations", "citations.corpusId", "citations.publicationDate"],
        batch_size=batch_size,
        progress_desc="Fetching citation lists",
    )

    for cid in missing:
        citation_cache[cid] = []

    for record in records:
        if record is None or record.get("corpusId") is None:
            continue
        focal_id = str(record["corpusId"])
        cits = []
        for cit in record.get("citations") or []:
            if cit is None or cit.get("corpusId") is None:
                continue
            cits.append((str(cit["corpusId"]), cit.get("publicationDate")))
        citation_cache[focal_id] = cits


def filter_citers_by_window(
    citer_records: List[Tuple[str, str]],
    focal_date: str,
    window_months: int,
) -> Set[str]:
    """Return citer corpus IDs whose publication date falls in [focal_date, focal_date + window]."""
    focal_dt = parse_date(focal_date)
    if focal_dt is None:
        return {cid for cid, _ in citer_records}

    if window_months is None or window_months <= 0:
        return {cid for cid, _ in citer_records}

    end_dt = focal_dt + datetime.timedelta(days=30 * window_months)
    filtered = set()
    for citer_id, citer_date in citer_records:
        citer_dt = parse_date(citer_date)
        if citer_dt is None:
            continue
        if focal_dt <= citer_dt <= end_dt:
            filtered.add(citer_id)
    return filtered


def compute_disruption_for_paper(
    focal_id: str,
    ref_ids: List[str],
    focal_date: str,
    citation_cache: Dict[str, List[Tuple[str, str]]],
    window_months: int,
):
    """Compute disruption index components and score for one focal paper."""
    focal_citers = filter_citers_by_window(citation_cache.get(focal_id, []), focal_date, window_months)

    ref_citers_union = set()
    for rid in ref_ids:
        ref_citers_union |= filter_citers_by_window(citation_cache.get(rid, []), focal_date, window_months)

    n_i = len(focal_citers - ref_citers_union)
    n_j = len(focal_citers & ref_citers_union)
    n_k = len(ref_citers_union - focal_citers)

    denom = n_i + n_j + n_k
    disruption = None if denom == 0 else (n_i - n_j) / float(denom)

    return {
        "n_i": n_i,
        "n_j": n_j,
        "n_k": n_k,
        "denominator": denom,
        "disruption_index": disruption,
        "num_focal_citers": len(focal_citers),
        "num_ref_citers_union": len(ref_citers_union),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute disruption index for PreScience target papers")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace dataset repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--window_months", type=int, default=12, help="Citation window in months from focal publication date; <=0 means unbounded")
    parser.add_argument("--max_key_references", type=int, default=10, help="Max key references per focal paper (for API cost control)")
    parser.add_argument("--batch_size", type=int, default=200, help="S2 batch size for /paper/batch calls")
    parser.add_argument("--limit", type=int, default=0, help="If >0, process only the first N target papers")
    parser.add_argument("--output_path", type=str, default="data/task_impact_prediction/test/disruption/disruption_index.json", help="Output JSON path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    utils.log(f"Loading corpus from HuggingFace: {args.hf_repo_id} (split={args.split})")
    all_papers, _, _ = utils.load_corpus(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embedding_type=None,
        load_sd2publications=False,
    )

    targets = [p for p in all_papers if "target" in p.get("roles", [])]
    targets = sorted(targets, key=lambda p: p.get("date", ""))
    if args.limit > 0:
        targets = targets[:args.limit]
    utils.log(f"Processing {len(targets)} target papers")

    results = []
    citation_cache: Dict[str, List[Tuple[str, str]]] = {}

    for paper in tqdm(targets, desc="Computing disruption index"):
        focal_id = paper["corpus_id"]
        focal_date = paper.get("date")
        key_refs = paper.get("key_references") or []
        ref_ids = [ref["corpus_id"] for ref in key_refs if isinstance(ref, dict) and ref.get("corpus_id")]
        if args.max_key_references > 0:
            ref_ids = ref_ids[:args.max_key_references]

        if len(ref_ids) == 0:
            results.append({
                "corpus_id": focal_id,
                "date": focal_date,
                "num_key_references_used": 0,
                "window_months": args.window_months,
                "n_i": 0,
                "n_j": 0,
                "n_k": 0,
                "denominator": 0,
                "disruption_index": None,
                "num_focal_citers": 0,
                "num_ref_citers_union": 0,
                "status": "no_key_references",
            })
            continue

        needed_ids = [focal_id] + ref_ids
        load_citation_lists(needed_ids, citation_cache, batch_size=args.batch_size)

        stats = compute_disruption_for_paper(
            focal_id=focal_id,
            ref_ids=ref_ids,
            focal_date=focal_date,
            citation_cache=citation_cache,
            window_months=args.window_months,
        )

        results.append({
            "corpus_id": focal_id,
            "date": focal_date,
            "num_key_references_used": len(ref_ids),
            "window_months": args.window_months,
            **stats,
            "status": "ok",
        })

    valid_scores = [r["disruption_index"] for r in results if r["disruption_index"] is not None]
    summary = {
        "num_targets": len(results),
        "num_with_valid_score": len(valid_scores),
        "mean_disruption_index": (sum(valid_scores) / len(valid_scores)) if valid_scores else None,
        "min_disruption_index": min(valid_scores) if valid_scores else None,
        "max_disruption_index": max(valid_scores) if valid_scores else None,
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    utils.log(f"Writing disruption results to {args.output_path}")
    utils.save_json(payload, args.output_path, metadata=utils.update_metadata([], args), overwrite=args.overwrite)
    utils.log("Done")


if __name__ == "__main__":
    main()
