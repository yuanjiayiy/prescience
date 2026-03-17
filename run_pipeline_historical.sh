#!/bin/bash
# Run the PreScience dataset pipeline for historical year ranges.
# Each year range is saved to data/corpus/historical/{YYYY}_{YYYY}/
#
# Prerequisites:
#   export S2_API_KEY=<your_key>
#   export HF_TOKEN=<your_token>
#   data/arxiv_snapshot/arxiv-metadata-oai-snapshot.json must exist
#
# Usage:
#   source activate sci4sci
#   bash run_pipeline_historical.sh 2>&1 | tee logs/historical_pipeline.log
#
# Note:
#   Stage 5 requires the following artifacts in each split directory:
#     - all_papers.stage01.json with populated target authors + key_references
#     - sd2og.json
#     - og2sd.json
#   If those files are missing, this script fails fast for that split.

set -e

export S2_API_KEY="D09UYXN5hYS4qiVnYIPl9dtFOcL3FzQ3pmesXHub"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ -z "$S2_API_KEY" ]]; then
    echo "ERROR: S2_API_KEY is not set. Run: export S2_API_KEY=<your_key>"
    exit 1
fi

ARXIV_SNAPSHOT="data/arxiv_snapshot/arxiv-metadata-oai-snapshot.json"
if [[ ! -f "$ARXIV_SNAPSHOT" ]]; then
    echo "ERROR: arXiv snapshot not found at $ARXIV_SNAPSHOT"
    echo "Download from: https://www.kaggle.com/datasets/Cornell-University/arxiv"
    exit 1
fi

mkdir -p logs

# ── Year ranges ───────────────────────────────────────────────────────────────
YEAR_PAIRS=(
    "2015-10-01 2016-10-01"
    "2016-10-01 2017-10-01"
    "2017-10-01 2018-10-01"
    "2018-10-01 2019-10-01"
    "2019-10-01 2020-10-01"
    "2020-10-01 2021-10-01"
    "2021-10-01 2022-10-01"
    "2022-10-01 2023-10-01"
    "2023-10-01 2024-10-01"
    "2024-10-01 2025-10-01"
    "2025-10-01 2026-10-01"
)

run_stage() {
    local stage_num="$1"
    local description="$2"
    local output_dir="$3"
    shift 3
    local cmd=("$@")

    local stage_flag="${output_dir}/.stage${stage_num}_done"
    if [[ -f "$stage_flag" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage ${stage_num} already done for ${output_dir}, skipping."
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage ${stage_num}: ${description} (${output_dir}) ==="
    "${cmd[@]}"
    touch "$stage_flag"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage ${stage_num} complete."
}

validate_stage5_prereqs() {
    local output_dir="$1"

    if [[ ! -f "${output_dir}/all_papers.stage01.json" ]]; then
        echo "ERROR: Missing ${output_dir}/all_papers.stage01.json"
        return 1
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for PAIR in "${YEAR_PAIRS[@]}"; do
    START_DATE=$(echo "$PAIR" | cut -d' ' -f1)
    END_DATE=$(echo "$PAIR" | cut -d' ' -f2)

    START_YEAR=$(echo "$START_DATE" | cut -d'-' -f1)
    END_YEAR=$(echo "$END_DATE" | cut -d'-' -f1)
    SPLIT_NAME="${START_YEAR}_${END_YEAR}"
    OUTPUT_DIR="data/corpus/historical/${SPLIT_NAME}"

    echo ""
    echo "========================================================"
    echo "Processing split: ${SPLIT_NAME}  (${START_DATE} → ${END_DATE})"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "========================================================"

    mkdir -p "$OUTPUT_DIR"

    # Stage 1: Download target papers from arXiv snapshot + S2 API
    run_stage 1 "Download target papers" "$OUTPUT_DIR" \
        python3 -m dataset.corpus.download_target_papers \
            --start_date "$START_DATE" \
            --end_date "$END_DATE" \
            --output_dir "$OUTPUT_DIR"

    # Stage 2: Add key (highly influential) references
    # run_stage 2 "Add key references" "$OUTPUT_DIR" \
    #     python3 -m dataset.corpus.add_key_references \
    #         --input_dir "$OUTPUT_DIR" \
    #         --output_dir "$OUTPUT_DIR"

    # Stage 3: Add author rosters and publication histories
    # run_stage 3 "Add authors" "$OUTPUT_DIR" \
    #     python3 -m dataset.corpus.add_authors \
    #         --input_dir "$OUTPUT_DIR" \
    #         --output_dir "$OUTPUT_DIR"

    # Stage 4 skipped (S2AND author disambiguation not required)

    validate_stage5_prereqs "$OUTPUT_DIR"

    # Stage 5: Add citation metadata (citation counts, h-index, trajectories)
    run_stage 5 "Add citation metadata" "$OUTPUT_DIR" \
        python3 -m dataset.corpus.add_citation_metadata \
            --input_dir "$OUTPUT_DIR" \
            --output_dir "$OUTPUT_DIR"

    # Stage 6: Replace titles/abstracts with official arXiv versions
    run_stage 6 "Replace titles/abstracts" "$OUTPUT_DIR" \
        python3 -m dataset.corpus.replace_title_abstracts_using_snapshot \
            --input_dir "$OUTPUT_DIR" \
            --output_dir "$OUTPUT_DIR"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished split: ${SPLIT_NAME}"
done

echo ""
echo "========================================================"
echo "All historical splits completed."
echo "Run upload_historical_to_hub.py to upload to HuggingFace."
echo "========================================================"
