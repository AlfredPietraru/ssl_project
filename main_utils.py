import logging
import os
import shutil
from wildlife_datasets.datasets import AnimalCLEF2026

from dotenv import load_dotenv
load_dotenv()
import kagglehub


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAIN")


def download_dataset():
    if os.path.isdir("data"):
        logger.info("`data` folder already exists. Skipping download.")
        return AnimalCLEF2026("data")
    
    logger.info("`data` folder not found. Downloading dataset...")
    try:
        path_of_files = kagglehub.competition_download("animal-clef-2026")
    except Exception:
        logger.exception("Dataset download failed in kagglehub.competition_download().")
        return None

    try:
        shutil.move(path_of_files, "data")
    except Exception:
        logger.exception("Dataset download succeeded, but moving files to `data` failed.")
        return
    logger.info("Dataset downloaded and moved to `data`.")
    return AnimalCLEF2026("data")


def download_mega_descriptor_model_feature_extraction():
    import timm
    try:
        m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    except Exception:
        logger.exception("Model for embeddings failed downloading...")
    m = m.eval()
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

