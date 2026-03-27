import re
import shutil
from pathlib import Path


INPUT_DIR = Path("logos")    # папка, где сейчас лежат картинки
OUTPUT_DIR = Path("dataset")     # куда собирать датасет

# Если True -> копировать файлы
# Если False -> перемещать файлы
COPY_FILES = True

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}



def clean_name(filename_stem: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", filename_stem).strip()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка INPUT_DIR: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Счётчик файлов внутри одного товарного знака
    counters = {}

    files = sorted([p for p in INPUT_DIR.iterdir() if is_image_file(p)])

    if not files:
        print("Входная папка пустая или в ней нет изображений.")
        return

    for file_path in files:
        original_stem = file_path.stem
        cleaned_stem = clean_name(original_stem)

        if not cleaned_stem:
            print(f"Не удалось получить имя: {file_path.name}")
            continue

        tm_folder = OUTPUT_DIR / cleaned_stem / "registry"
        tm_folder.mkdir(parents=True, exist_ok=True)

        # нумерация
        counters.setdefault(cleaned_stem, 0)
        counters[cleaned_stem] += 1
        file_number = counters[cleaned_stem]

        new_name = f"reg_{file_number:02d}{file_path.suffix.lower()}"
        destination = tm_folder / new_name

        if COPY_FILES:
            shutil.copy2(file_path, destination)
        else:
            shutil.move(str(file_path), str(destination))

        print(f"OK {file_path.name} -> {destination}")



if __name__ == "__main__":
    main()