from pathlib import Path

DATASET_DIR = Path("dataset")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Папка не найдена: {DATASET_DIR}")

    renamed_count = 0

    # ищем папки вида dataset/<tm_id>/real
    for tm_folder in sorted(DATASET_DIR.iterdir()):
        if not tm_folder.is_dir():
            continue

        real_folder = tm_folder / "real"
        if not real_folder.exists() or not real_folder.is_dir():
            continue

        image_files = sorted([p for p in real_folder.iterdir() if is_image_file(p)])

        if not image_files:
            continue

        # делаем временные имена, чтобы избежать конфликтов
        temp_files = []
        for i, file_path in enumerate(image_files, start=1):
            temp_name = f"__tmp_real_{i:03d}{file_path.suffix.lower()}"
            temp_path = real_folder / temp_name
            file_path.rename(temp_path)
            temp_files.append(temp_path)

        # переименовываем
        for i, temp_path in enumerate(sorted(temp_files), start=1):
            new_name = f"real_{i:02d}{temp_path.suffix.lower()}"
            new_path = real_folder / new_name
            temp_path.rename(new_path)
            renamed_count += 1
            print(f"OK {tm_folder.name}: {new_name}")

    print(f"\nПереименовано файлов: {renamed_count}")


if __name__ == "__main__":
    main()