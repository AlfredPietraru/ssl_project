#!/usr/bin/env bash
set -euo pipefail

source venv_ssl_proj/bin/activate
# python3 main_utils.py
# python3 main.py
python3 02_extract_embeddings.py
python3 03_04_build_gallery_and_validation.py
python3 05_calibrate_rejection_thresholds.py
python3 06_run_nearest_neighbor_matching.py
python3 07_reject_low_confidence_matches.py
python3 08_cluster_rejected_unknowns.py
python3 09_refine_with_incremental_enrollment.py
python3 10_export_final_assignments.py
