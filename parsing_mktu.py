import csv
import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup


POST_URL = "https://www.fips.ru/registers-doc-view/fips_servlet"
INPUT_TXT = "retry_result.txt"
OUTPUT_JSON = "inn_to_tm_mktu_2.json"
OUTPUT_CSV = "progress_2.csv"
DEBUG_FILE = "debug_not_found.txt"

SLEEP = 3.0
RETRY_SLEEP = 3
MAX_RETRIES = 3
SAVE_JSON_EVERY = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.fips.ru/",
}


def load_inn_dict(path):
    result = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            inn, nums = line.split(":", 1)
            numbers = [x.strip() for x in nums.split(",") if x.strip()]
            result[inn.strip()] = numbers

    return result


def load_existing_json(path):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_csv():
    if os.path.exists(OUTPUT_CSV):
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["inn", "reg_number", "mktu", "status"])


def append_csv(inn, reg, mktu, status="ok"):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([inn, reg, ",".join(map(str, mktu)), status])


def init_debug_file():
    if os.path.exists(DEBUG_FILE):
        return

    with open(DEBUG_FILE, "w", encoding="utf-8") as f:
        f.write("DEBUG NOT FOUND\n\n")


def fetch_html_once(session, reg):
    data = {
        "DB": "RUTM",
        "DocNumber": reg,
        "TypeFile": "html",
        "searchPar": "par_1",
        "searchParValue": reg
    }

    r = session.post(POST_URL, headers=HEADERS, data=data, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def is_rate_limited(html):
    return "Слишком быстрый просмотр документов" in html


def fetch_html_with_retry(session, reg):
    last_html = None

    for attempt in range(1, MAX_RETRIES + 1):
        html = fetch_html_once(session, reg)
        last_html = html

        if not is_rate_limited(html):
            return html, False, attempt

        print(f"   -> BLOCKED, попытка {attempt}/{MAX_RETRIES}, ждём {RETRY_SLEEP} сек")
        time.sleep(RETRY_SLEEP)

    return last_html, True, MAX_RETRIES


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n", strip=True)


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_block_by_code(text, code):
    patterns = [
        rf"\({code}\)(.*?)(?=\(\d{{3}}\)|$)",
        rf"{code}\)(.*?)(?=\(\d{{3}}\)|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return normalize_text(match.group(1))

    return None


def extract_block_511(text):
    candidates = []

    block = extract_block_by_code(text, "511")
    if block:
        candidates.append(block)

    title_patterns = [
        r"Классы МКТУ.*?(?=\(\d{3}\)|$)",
        r"Классы товаров и/или услуг.*?(?=\(\d{3}\)|$)",
        r"Перечень товаров и/или услуг.*?(?=\(\d{3}\)|$)",
    ]

    for pattern in title_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            candidates.append(normalize_text(match.group(0)))

    idx = text.find("(511)")
    if idx != -1:
        fragment = text[idx: idx + 4000]
        candidates.append(normalize_text(fragment))

    for c in candidates:
        if c:
            return c

    return None


def extract_mktu(block):
    if not block:
        return []

    # убираем лишние пробелы и переносы
    block = re.sub(r"\s+", " ", block)

    # ищем ВСЕ числа от 1 до 45 (включая 01, 07 и т.д.)
    found = re.findall(r"\b0?([1-9]|[1-3]\d|4[0-5])\b", block)

    classes = sorted({int(x) for x in found})

    # защита от мусора (если вдруг нашли слишком много чисел)
    if len(classes) > 45:
        return []

    return classes


def write_debug(reg, inn, text, block_511):
    with open(DEBUG_FILE, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ИНН: {inn}\n")
        f.write(f"REG: {reg}\n\n")
        f.write("BLOCK_511:\n")
        f.write((block_511 or "None") + "\n\n")
        f.write("TEXT_HEAD:\n")
        f.write(text[:3000] + "\n\n")


def count_total_numbers(inn_numbers):
    return sum(len(v) for v in inn_numbers.values())


def count_processed(existing_result, inn_numbers):
    count = 0
    for inn, numbers in inn_numbers.items():
        for reg in numbers:
            if inn in existing_result and reg in existing_result[inn]:
                count += 1
    return count


def main():
    inn_numbers = load_inn_dict(INPUT_TXT)

    init_csv()
    init_debug_file()

    result = load_existing_json(OUTPUT_JSON)

    total = count_total_numbers(inn_numbers)
    current = count_processed(result, inn_numbers)

    print(f"Всего номеров: {total}")
    print(f"Уже обработано: {current}")
    print(f"Осталось: {total - current}")

    start_time = time.time()

    with requests.Session() as session:
        for inn, numbers in inn_numbers.items():
            if inn not in result:
                result[inn] = {}

            print(f"\nИНН {inn}")

            for reg in numbers:
                if reg in result[inn]:
                    print(f"[SKIP] уже есть | {inn} | {reg} -> {result[inn][reg]}")
                    continue

                current += 1
                percent = round(current / total * 100, 2)
                remaining = total - current

                elapsed = time.time() - start_time
                avg_time = elapsed / max(current, 1)
                eta = int(avg_time * remaining)

                print(f"[{current}/{total}] ({percent}%) осталось: {remaining} | ETA: {eta} сек | {reg}")

                try:
                    html, blocked, attempts = fetch_html_with_retry(session, reg)

                    if blocked:
                        result[inn][reg] = []
                        append_csv(inn, reg, [], status="blocked")
                        print("   -> ERROR: ФИПС вернул ограничение после всех попыток")
                    else:
                        text = html_to_text(html)
                        block_511 = extract_block_511(text)
                        mktu = extract_mktu(block_511)

                        result[inn][reg] = mktu

                        if mktu:
                            append_csv(inn, reg, mktu, status="ok")
                            print(f"   -> {mktu}")
                        else:
                            append_csv(inn, reg, [], status="not_found")
                            print("   -> [] (511 не распарсился)")
                            write_debug(reg, inn, text, block_511)

                except Exception as e:
                    result[inn][reg] = []
                    append_csv(inn, reg, [], status=f"error: {e}")
                    print(f"   -> ERROR {e}")

                if current % SAVE_JSON_EVERY == 0:
                    save_json(result)
                    print("   -> промежуточное сохранение JSON")

                time.sleep(SLEEP)

    save_json(result)

    print("\nDONE")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"DEBUG: {DEBUG_FILE}")


if __name__ == "__main__":
    main()