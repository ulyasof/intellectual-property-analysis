from config import OUTPUT_DIR
from utils import read_csv, read_txt, write_csv
# неактуальный

def main():
    images_metadata_path = OUTPUT_DIR / "images_metadata.csv"
    train_ids_path = OUTPUT_DIR / "splits" / "train_ids.txt"
    val_ids_path = OUTPUT_DIR / "splits" / "val_ids.txt"
    test_ids_path = OUTPUT_DIR / "splits" / "test_ids.txt"

    image_rows = read_csv(images_metadata_path)

    train_ids = set(read_txt(train_ids_path))
    val_ids = set(read_txt(val_ids_path))
    test_ids = set(read_txt(test_ids_path))

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

    write_csv(
        OUTPUT_DIR / "images_metadata_with_split.csv",
        fieldnames=[
            "image_id",
            "tm_id",
            "reg_number",
            "path",
            "source_type",
            "is_real",
            "is_augmented",
            "variant_status",
            "split",
        ],
        rows=new_rows
    )

    print(f"Готово: {OUTPUT_DIR / 'images_metadata_with_split.csv'}")


if __name__ == "__main__":
    main()