#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="logs.txt"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

log_step() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step() {
    local step_name="${1:-}"
    if [[ -z "$step_name" ]]; then
        log_step "FAILED run_step called without a script name"
        return 2
    fi

    log_step "START ${step_name}"
    if python3 "$step_name"; then
        log_step "END ${step_name}"
    else
        local exit_code=$?
        log_step "FAILED ${step_name} exit_code=${exit_code}"
        return "$exit_code"
    fi
}

source venv_ssl_proj/bin/activate
log_step "Pipeline started"
# python3 main_utils.py
python3 main.py
run_step 02_extract_embeddings.py
run_step 03_cluster_test_embeddings.py

# Optional lookup + discovery pipeline. This path is experimental and is not
# required for the direct ARI clustering submission.
# run_step 03_04_build_gallery_and_validation.py
# run_step 05_calibrate_rejection_thresholds.py
# run_step 06_run_nearest_neighbor_matching.py
# run_step 07_reject_low_confidence_matches.py
# run_step 08_cluster_rejected_unknowns.py
# run_step 09_refine_with_incremental_enrollment.py
# run_step 10_export_final_assignments.py
run_step 11_verify_train_clustering.py
log_step "Pipeline finished"
