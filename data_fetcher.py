from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")


class DataFetcher:
    def __init__(self, db_path: str | Path = "metadata.sqlite") -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

    def execute_query(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cursor = self.connection.cursor() # type: ignore
        if params is None:
            cursor.execute(query)
        else:
            cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_distinct_animals_split(self, split : str) -> list[str]:
        query = f"""
        SELECT
            DISTINCT m.species
        FROM metadata m
        WHERE m.split = '{split}'
        """
        res = self.execute_query(query)
        elems = list({r.get("species", "") for r in res})
        elems.sort()
        return elems

    def get_train_split_animal(
        self,
        animal: str,
        split_size: float = 0.8,
        random_seed: int = 42,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not 0.0 < split_size < 1.0:
            raise ValueError("split_size must be strictly between 0 and 1.")
        query = """
        SELECT
            m.identity as identity,
            json_group_array(m.path) AS paths
        FROM metadata m
        WHERE m.split = 'train' and m.species = ?
        GROUP BY m.identity
        """
        rows = self.execute_query(query, (animal,))
        identity_to_int = {
            str(row["identity"]): index
            for index, row in enumerate(rows)
        }
        for row in rows:
            row["identity"] = identity_to_int[str(row["identity"])]
            row["paths"] = json.loads(row["paths"]) if row.get("paths") else []

        if not rows:
            return [], []

        rng = random.Random(random_seed)
        shuffled_rows = rows[:]
        rng.shuffle(shuffled_rows)

        split_index = int(len(shuffled_rows) * split_size)
        if len(shuffled_rows) > 1:
            split_index = max(1, min(split_index, len(shuffled_rows) - 1))

        train_rows = shuffled_rows[:split_index]
        only_one_entry_train = 0
        for row in train_rows:
            only_one_entry_train = only_one_entry_train + 1 if len(row["paths"]) == 1 else only_one_entry_train
        logger.warning(f"There are {only_one_entry_train} entities in train with 1 example.")
        train_rows = list(filter(lambda x: len(x.get("paths", 0)) > 1, train_rows))

        val_rows = shuffled_rows[split_index:]
        only_one_entry_val = 0
        for row in val_rows:
            only_one_entry_val = only_one_entry_val + 1 if len(row["paths"]) == 1 else only_one_entry_val
        logger.warning(f"There are {only_one_entry_val} entities in val with 1 example.")

        return train_rows, val_rows


    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if hasattr(self, "connection") and self.connection is not None:
            self.connection.close()
            self.connection = None

    

if __name__ == "__main__":
    fetcher = DataFetcher()
    # animals = fetcher.get_distinct_animals_split(split='train')
    # print(animals)
    train_res, val_res = fetcher.get_train_split_animal(animal="salamander")
    print("Train results:", len(train_res))
    print("Validation results:", len(val_res))
    # print(json.dumps(train_res[:3], indent=2))
    # print(json.dumps(val_res[:3], indent=2))
    # print(len(train_res), len(val_res))
