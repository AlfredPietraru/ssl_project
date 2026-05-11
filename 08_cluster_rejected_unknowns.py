import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from main_utils import normalize_rows

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP08_CLUSTERING")


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    n = adjacency.shape[0]
    visited = np.zeros(n, dtype=bool)
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = np.flatnonzero(adjacency[node])
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        components.append(component)
    return components


def cluster_rejected_unknowns(cfg: CFG) -> None:
    rejection_dir = Path(cfg.rejection_output_dir)
    embeddings_dir = Path(cfg.embeddings_output_dir)
    output_dir = Path(cfg.clustering_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rejected = pd.read_csv(rejection_dir / "rejected_unknowns.csv")
    test_embeddings = np.load(embeddings_dir / "test_embeddings.npy")
    test_metadata = pd.read_csv(embeddings_dir / "test_metadata.csv")
    test_embeddings = normalize_rows(test_embeddings)

    if rejected.empty:
        logger.info("No rejected unknown queries were found. Writing empty clustering artifacts.")
        pd.DataFrame().to_csv(output_dir / "unknown_cluster_assignments.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "unknown_cluster_prototypes_metadata.csv", index=False)
        np.save(output_dir / "unknown_cluster_prototypes_embeddings.npy", np.empty((0, 0), dtype=np.float32))
        return

    assignments: list[dict[str, object]] = []
    prototype_rows: list[dict[str, object]] = []
    prototype_vectors: list[np.ndarray] = []

    if "query_species" in rejected.columns:
        species_groups = list(rejected.groupby("query_species", sort=True))
    else:
        species_groups = [("all", rejected)]

    for species_name, group in species_groups:
        query_indices = group["query_index"].to_numpy(dtype=int)
        group_embeddings = test_embeddings[query_indices]
        similarity_matrix = group_embeddings @ group_embeddings.T
        adjacency = similarity_matrix >= cfg.clustering_similarity_threshold
        components = connected_components(adjacency)

        cluster_counter = 0
        for component in components:
            if len(component) < cfg.clustering_min_cluster_size:
                continue

            cluster_counter += 1
            member_positions = np.array(component, dtype=int)
            member_query_indices = query_indices[member_positions]
            member_embeddings = group_embeddings[member_positions]
            prototype = normalize_rows(member_embeddings.mean(axis=0, keepdims=True))[0]
            prototype_vectors.append(prototype)

            if species_name == "all" or pd.isna(species_name):
                cluster_id = f"new_identity_{cluster_counter:04d}"
            else:
                cluster_id = f"new_{str(species_name)}_{cluster_counter:04d}"

            component_similarity = similarity_matrix[np.ix_(member_positions, member_positions)]
            internal_similarity = float(component_similarity.mean())

            prototype_rows.append(
                {
                    "cluster_id": cluster_id,
                    "prototype_index": len(prototype_vectors) - 1,
                    "cluster_size": int(len(member_query_indices)),
                    "mean_internal_similarity": internal_similarity,
                    "species": species_name,
                }
            )

            for query_index in member_query_indices:
                query_row = test_metadata.iloc[int(query_index)]
                row: dict[str, object] = {
                    "query_index": int(query_index),
                    "cluster_id": cluster_id,
                    "cluster_size": int(len(member_query_indices)),
                    "mean_internal_similarity": internal_similarity,
                }
                for candidate in ["identity", "species", "dataset", "image_id", "path"]:
                    if candidate in test_metadata.columns:
                        row[f"query_{candidate}"] = query_row[candidate]
                assignments.append(row)

    assignments_df = pd.DataFrame(assignments)
    prototypes_df = pd.DataFrame(prototype_rows)
    prototype_embeddings = (
        np.stack(prototype_vectors, axis=0)
        if prototype_vectors
        else np.empty((0, test_embeddings.shape[1]), dtype=np.float32)
    )

    assignments_df.to_csv(output_dir / "unknown_cluster_assignments.csv", index=False)
    prototypes_df.to_csv(output_dir / "unknown_cluster_prototypes_metadata.csv", index=False)
    np.save(output_dir / "unknown_cluster_prototypes_embeddings.npy", prototype_embeddings)

    summary = {
        "num_rejected_queries": int(len(rejected)),
        "num_clusters": int(len(prototypes_df)),
        "clustering_similarity_threshold": cfg.clustering_similarity_threshold,
        "clustering_min_cluster_size": cfg.clustering_min_cluster_size,
    }
    with (output_dir / "clustering_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    logger.info("Saved unknown cluster assignments to %s", output_dir / "unknown_cluster_assignments.csv")
    logger.info("Saved unknown cluster prototypes to %s", output_dir / "unknown_cluster_prototypes_metadata.csv")


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    cluster_rejected_unknowns(cfg)
