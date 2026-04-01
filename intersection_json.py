import json

# Пути к файлам
file1 = 'inn_to_tm_mktu.json'    # основной файл
file2 = 'inn_to_tm_mktu_2.json'   # файл с дозаполнением
output_file = 'merged.json'

with open(file1, 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open(file2, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

for inn, marks2 in data2.items():
    # если ИНН вообще не было в первом файле — добавляем целиком
    if inn not in data1:
        data1[inn] = marks2
        continue

    # иначе проходим по товарным знакам
    for reg_num, classes2 in marks2.items():
        # если товарного знака не было в первом файле — добавляем
        if reg_num not in data1[inn]:
            data1[inn][reg_num] = classes2
        else:
            classes1 = data1[inn][reg_num]

            # если в первом файле пропуск (пустой список), заменяем данными из второго
            if classes1 == [] and classes2 != []:
                data1[inn][reg_num] = classes2

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data1, f, ensure_ascii=False, indent=2)

print(f'Готово: результат сохранён в {output_file}')