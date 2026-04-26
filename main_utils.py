import logging
import os
import shutil
from pathlib import Path

from wildlife_datasets.datasets import AnimalCLEF2026

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))
import kagglehub
from kagglehub.config import get_kaggle_credentials
from kagglehub.exceptions import KaggleApiHTTPError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAIN")


def _log_kaggle_auth_state() -> None:
    credentials = get_kaggle_credentials()
    has_token_env = bool(os.getenv("KAGGLE_API_TOKEN"))
    has_username_env = bool(os.getenv("KAGGLE_USERNAME"))
    has_key_env = bool(os.getenv("KAGGLE_KEY"))

    logger.info(
        "Kaggle auth loaded: credentials_present=%s, token_env=%s, username_env=%s, key_env=%s",
        bool(credentials),
        has_token_env,
        has_username_env,
        has_key_env,
    )


def download_dataset():
    if os.path.isdir("data"):
        logger.info("`data` folder already exists. Skipping download.")
        return AnimalCLEF2026("data")

    logger.info("`data` folder not found. Downloading dataset...")
    _log_kaggle_auth_state()

    try:
        kagglehub.competition_download("animal-clef-2026", output_dir="data")
    except KaggleApiHTTPError as exc:
        logger.exception("Dataset download failed in kagglehub.competition_download().")
        if exc.response is not None and exc.response.status_code == 401:
            logger.error(
                "Kaggle returned 401. This usually means one of these is true: "
                "(1) `KAGGLE_API_TOKEN` was not loaded into this process, "
                "(2) the token is invalid or belongs to a different account, or "
                "(3) that Kaggle account has not accepted the AnimalCLEF 2026 competition rules yet."
            )
            logger.error(
                "Open https://www.kaggle.com/competitions/animal-clef-2026/rules while logged into "
                "the same Kaggle account as the API token and accept the rules, then retry."
            )
        return None
    except Exception:
        logger.exception("Dataset download failed in kagglehub.competition_download().")
        return None

    logger.info("Dataset downloaded and moved to `data`.")
    return AnimalCLEF2026("data")


def download_mega_descriptor_model_feature_extraction(eval = True):
    import timm
    try:
        m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    except Exception:
        logger.exception("Model for embeddings failed downloading...")
        raise
    if eval:
        m = m.eval()
    else:
        m = m.train()
    return m

def download_wildlife_pretraining():
    try:
        path = kagglehub.dataset_download("wildlifedatasets/wildlifereid-10k")
        shutil.move(path, "pretrained_data")
    except Exception:
        logger.exception("Failed downloading other moving dataset to the right path")
    

if __name__ == "__main__":
    dataset = download_dataset()
    if dataset is None:
        print("empty dataset")
        exit(1)
    print(dataset.metadata)
