import json

INPUT_JSON = "inn_to_tm_mktu.json"
OUTPUT_TXT = "retry_result.txt"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

retry_data = {}

for inn, marks in data.items():
    missing_regs = []

    for reg_number, mktu in marks.items():
        if not mktu:   # ловит []
            missing_regs.append(str(reg_number))

    if missing_regs:
        retry_data[str(inn)] = missing_regs

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for inn, reg_numbers in retry_data.items():
        f.write(f"{inn}: {', '.join(reg_numbers)}\n")

print("Готово.")
print(f"Сохранено в {OUTPUT_TXT}")
print(f"ИНН с пропусками: {len(retry_data)}")
print(f"Всего пропущенных номеров: {sum(len(v) for v in retry_data.values())}")