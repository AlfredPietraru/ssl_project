from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    root = Path("data/images")
    total_train = 0
    total_test = 0

    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        train_count = count_images(dataset_dir / "train")
        test_count = count_images(dataset_dir / "test")
        total_train += train_count
        total_test += test_count
        print(f"{dataset_dir.name}: train={train_count}, test={test_count}")

    print(f"TOTAL: train={total_train}, test={total_test}")


if __name__ == "__main__":
    main()
