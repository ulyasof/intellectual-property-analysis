from pathlib import Path
import csv

IMAGES_METADATA_INPUT = Path("prepared/images_metadata.csv")
TM_METADATA_INPUT = Path("prepared/tm_metadata_filled.csv")
OUTPUT_PATH = Path("prepared/images_metadata_with_mktu.csv")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not IMAGES_METADATA_INPUT.exists():
        raise FileNotFoundError(f"Не найден файл: {IMAGES_METADATA_INPUT}")

    if not TM_METADATA_INPUT.exists():
        raise FileNotFoundError(f"Не найден файл: {TM_METADATA_INPUT}")

    image_rows = read_csv(IMAGES_METADATA_INPUT)
    tm_rows = read_csv(TM_METADATA_INPUT)

    tm_map = {}
    for row in tm_rows:
        tm_id = str(row.get("tm_id", "")).strip()
        tm_map[tm_id] = {
            "brand_name": row.get("brand_name", "").strip(),
            "mktu_classes": row.get("mktu_classes", "").strip(),
            "status": row.get("status", "").strip(),
        }

    merged_rows = []
    for row in image_rows:
        tm_id = str(row.get("tm_id", "")).strip()
        extra = tm_map.get(tm_id, {"brand_name": "", "mktu_classes": "", "status": ""})

        new_row = dict(row)
        new_row["brand_name"] = extra["brand_name"]
        new_row["mktu_classes"] = extra["mktu_classes"]
        new_row["status"] = extra["status"]

        merged_rows.append(new_row)

    fieldnames = list(merged_rows[0].keys()) if merged_rows else [
        "image_id", "tm_id", "reg_number", "path", "source_type",
        "is_real", "is_augmented", "variant_status",
        "brand_name", "mktu_classes", "status"
    ]

    write_csv(OUTPUT_PATH, fieldnames, merged_rows)
    print(f"Готово: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()