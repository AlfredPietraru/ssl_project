import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from main_utils import require_existing_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP07_REJECTION")


def reject_low_confidence_matches(cfg: CFG) -> None:
    matching_dir = Path(cfg.matching_output_dir)
    thresholds_dir = Path(cfg.thresholds_output_dir)
    output_dir = Path(cfg.rejection_output_dir)
    require_existing_paths(
        [
            (matching_dir / "match_scores.csv", "step 06 nearest-neighbor matching"),
            (thresholds_dir / "rejection_thresholds.json", "step 05 threshold calibration"),
        ],
        step_name="Step 07 low-confidence rejection",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(matching_dir / "match_scores.csv")
    with (thresholds_dir / "rejection_thresholds.json").open("r", encoding="utf-8") as handle:
        thresholds = json.load(handle)

    similarity_threshold = float(thresholds["similarity_threshold"])
    margin_threshold = float(thresholds["margin_threshold"])

    scores["accepted_known"] = (
        scores["top1_similarity"].to_numpy(dtype=float) >= similarity_threshold
    ) & (
        np.where(
            np.isfinite(scores["top1_top2_margin"].to_numpy(dtype=float)),
            scores["top1_top2_margin"].to_numpy(dtype=float) >= margin_threshold,
            True,
        )
    )

    scores["rejection_reason"] = ""
    low_similarity = scores["top1_similarity"] < similarity_threshold
    low_margin = np.isfinite(scores["top1_top2_margin"]) & (scores["top1_top2_margin"] < margin_threshold)
    scores.loc[~scores["accepted_known"] & low_similarity, "rejection_reason"] = "low_similarity"
    scores.loc[~scores["accepted_known"] & low_margin, "rejection_reason"] = "low_margin"

    known_matches = scores[scores["accepted_known"]].reset_index(drop=True).copy()
    rejected_unknowns = scores[~scores["accepted_known"]].reset_index(drop=True).copy()

    scores.to_csv(output_dir / "rejection_decisions.csv", index=False)
    known_matches.to_csv(output_dir / "known_matches.csv", index=False)
    rejected_unknowns.to_csv(output_dir / "rejected_unknowns.csv", index=False)

    logger.info("Saved rejection decisions to %s", output_dir / "rejection_decisions.csv")
    logger.info("Saved accepted known matches to %s", output_dir / "known_matches.csv")
    logger.info("Saved rejected unknown queries to %s", output_dir / "rejected_unknowns.csv")


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    reject_low_confidence_matches(cfg)
