from config import DATASET_DIR, OUTPUT_DIR
from utils import is_image_file, write_csv, write_txt


def collect_dataset_info(dataset_dir):
    image_rows = []
    tm_rows = []
    tm_ids = []

    image_id = 1

    for tm_folder in sorted(dataset_dir.iterdir()):
        if not tm_folder.is_dir():
            continue

        tm_id = tm_folder.name.strip()
        reg_number = tm_id

        if not tm_id:
            continue

        tm_ids.append(tm_id)

        registry_dir = tm_folder / "registry"
        if registry_dir.exists() and registry_dir.is_dir():
            registry_files = sorted([p for p in registry_dir.iterdir() if is_image_file(p)])
            for file_path in registry_files:
                image_rows.append({
                    "image_id": image_id,
                    "tm_id": tm_id,
                    "reg_number": reg_number,
                    "path": str(file_path.as_posix()),
                    "source_type": "registry",
                    "is_real": 0,
                    "is_augmented": 0,
                    "variant_status": "main",
                })
                image_id += 1

        real_dir = tm_folder / "real"
        if real_dir.exists() and real_dir.is_dir():
            real_files = sorted([p for p in real_dir.iterdir() if is_image_file(p)])
            for file_path in real_files:
                image_rows.append({
                    "image_id": image_id,
                    "tm_id": tm_id,
                    "reg_number": reg_number,
                    "path": str(file_path.as_posix()),
                    "source_type": "real",
                    "is_real": 1,
                    "is_augmented": 0,
                    "variant_status": "main",
                })
                image_id += 1

        tm_rows.append({
            "tm_id": tm_id,
            "reg_number": reg_number,
            "brand_name": "",
            "mktu_classes": "",
            "status": "",
        })

    return image_rows, tm_rows, tm_ids


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка: {DATASET_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_rows, tm_rows, tm_ids = collect_dataset_info(DATASET_DIR)

    write_csv(
        OUTPUT_DIR / "images_metadata.csv",
        fieldnames=[
            "image_id",
            "tm_id",
            "reg_number",
            "path",
            "source_type",
            "is_real",
            "is_augmented",
            "variant_status",
        ],
        rows=image_rows
    )

    write_csv(
        OUTPUT_DIR / "tm_metadata_template.csv",
        fieldnames=[
            "tm_id",
            "reg_number",
            "brand_name",
            "mktu_classes",
            "status",
        ],
        rows=tm_rows
    )

    write_txt(OUTPUT_DIR / "all_tm_ids.txt", tm_ids)

    print(f"Товарных знаков: {len(tm_ids)}")
    print(f"Изображений: {len(image_rows)}")
    print(f"Создано: {OUTPUT_DIR / 'images_metadata.csv'}")
    print(f"Создано: {OUTPUT_DIR / 'tm_metadata_template.csv'}")
    print(f"Создано: {OUTPUT_DIR / 'all_tm_ids.txt'}")


if __name__ == "__main__":
    main()