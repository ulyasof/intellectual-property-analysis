from pathlib import Path
import csv

INPUT_CSV = Path("prepared/images_metadata_with_mktu.csv")
TRAIN_IDS = Path("prepared/splits/train_ids.txt")
VAL_IDS = Path("prepared/splits/val_ids.txt")
TEST_IDS = Path("prepared/splits/test_ids.txt")
OUTPUT_CSV = Path("prepared/images_metadata_final.csv")


def read_txt(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
    image_rows = read_csv(INPUT_CSV)

    train_ids = set(read_txt(TRAIN_IDS))
    val_ids = set(read_txt(VAL_IDS))
    test_ids = set(read_txt(TEST_IDS))

    new_rows = []
    for row in image_rows:
        tm_id = row["tm_id"]

        if tm_id in train_ids:
            row["split"] = "train"
        elif tm_id in val_ids:
            row["split"] = "val"
        elif tm_id in test_ids:
            row["split"] = "test"
        else:
            row["split"] = "unknown"

        new_rows.append(row)

    fieldnames = list(new_rows[0].keys())
    write_csv(OUTPUT_CSV, fieldnames, new_rows)

    print(f"Готово: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()