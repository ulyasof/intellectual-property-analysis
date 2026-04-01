import json
import csv

input_file = 'merged.json'
output_file = 'result_mktu.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # заголовок
    writer.writerow(['inn', 'reg_num', 'classes'])

    for inn, marks in data.items():
        for reg_num, classes in marks.items():
            # превращаем список классов в строку
            classes_str = ','.join(map(str, classes)) if classes else ''
            writer.writerow([inn, reg_num, classes_str])

print('Готово')