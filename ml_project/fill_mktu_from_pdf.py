from pathlib import Path
import csv
import re

from pypdf import PdfReader

PDF_PATH = Path("классы мкту .pdf")
TM_METADATA_INPUT = Path("prepared/tm_metadata_template.csv")
TM_METADATA_OUTPUT = Path("prepared/tm_metadata_filled.csv")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_reg_to_mktu_from_text(text: str) -> dict[str, str]:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    pattern = re.compile(
        r"Номер регистрации:\s*(\d{6,7})\s*"
        r"Классы мкту:\s*([0-9\s]+)",
        flags=re.IGNORECASE
    )

    reg_to_mktu = {}

    for match in pattern.finditer(text):
        reg_number = match.group(1).strip()
        raw_classes = match.group(2).strip()

        classes = re.findall(r"\d{1,2}", raw_classes)
        classes = [c.zfill(2) for c in classes]

        if classes:
            reg_to_mktu[reg_number] = " ".join(classes)

    return reg_to_mktu


def extract_reg_to_mktu(pdf_path: Path) -> dict[str, str]:
    reader = PdfReader(str(pdf_path))
    full_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)

    joined_text = "\n".join(full_text)
    return extract_reg_to_mktu_from_text(joined_text)


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
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF не найден: {PDF_PATH}")

    if not TM_METADATA_INPUT.exists():
        raise FileNotFoundError(f"CSV не найден: {TM_METADATA_INPUT}")

    reg_to_mktu = extract_reg_to_mktu(PDF_PATH)
    print(f"Из pdf извлечено записей: {len(reg_to_mktu)}")

    rows = read_csv(TM_METADATA_INPUT)

    updated = 0
    missed = []

    for row in rows:
        reg_number = str(row.get("reg_number", "")).strip()
        tm_id = str(row.get("tm_id", "")).strip()

        key = reg_number if reg_number else tm_id

        if key in reg_to_mktu:
            row["mktu_classes"] = reg_to_mktu[key]
            updated += 1
        else:
            missed.append(key)

    fieldnames = list(rows[0].keys()) if rows else [
        "tm_id", "reg_number", "brand_name", "mktu_classes", "status"
    ]

    write_csv(TM_METADATA_OUTPUT, fieldnames, rows)

    print(f"Обновлено строк: {updated}")
    print(f"Сохранено: {TM_METADATA_OUTPUT}")

    if missed:
        print("\nНе найдены в PDF:")
        for x in missed[:30]:
            print(" -", x)
        if len(missed) > 30:
            print(f"{len(missed) - 30}")


if __name__ == "__main__":
    main()