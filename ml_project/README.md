Проект для поиска визуально похожих товарных знаков по изображению и оценка риска.

## Что делает проект 

1. Поиск похожиъ товарных знаков по входному изображению
2. Оценка риска конфликта, которая учитывает визуальное сходство, пересечение классов МКТУ

## Основные функции 

- подготовка датасета 
- обучение retrieval модели 
- инференс по одной картинке
- возврат пяти похожих логотипов
- вывод классов МКТУ для знаков 
- расчет risk_score
- поддерживает отдельный эксперимент с `hard negatives`

## Основные файлы 

- inference_search.py - по картинке ищет похожие товарные знаки 
- risk_utils.py - логика расчета риска
- retrieval_dataset.py - общий модуль данных для обучения модели
- train_metric.py - обучение модели по фолдам
- train_final_model.py - обучение финальной модели
- evaluate_retrieval.py - оценка качества 
- config.py - настройки и пути
- utils.py - вспомогательные функции

Все команды ниже предполагают, что запуск идёт **из папки `ml_project/`**:

## Подготовка данных 

- organize_fips.py - раскладывает изображения по структуре `dataset/<tm_id>/registry`
- rename_real.py — переименовывает реальные изображения в папках `real`
- scan_dataset.py — собирает `images_metadata.csv` и `tm_metadata_template.csv`
- fill_mktu_from_pdf.py — извлекает классы МКТУ из PDF и сохраняет `tm_metadata_filled.csv`
- merge_mktu_into_images.py — добавляет МКТУ и статус к метаданным изображений
- split_to_images.py — добавляет колонку `split` в итоговый CSV
- build_retrieval_folds.py — строит CV-фолды для retrieval-задачи
- build_final_retrieval_set.py — строит финальный retrieval-набор


## Обучение 
1. Подготавливаются изображения 
2. Собираются метаданные по изображениям 
3. Из pdf берет классы МКТУ
4. Формируются пары
5. Обучается модель 
6. Финальная модель сохраняет
    - best_model.pt
    - gallery_embeddings.pt
    - gallery_metadata.csv
  
#### Обучение и оценка
- retrieval_dataset.py — базовый dataset для обучения
- train_metric.py — основное CV-обучение retrieval-модели
- evaluate_retrieval.py — zero-shot retrieval оценка по fold-ам
- train_final_model.py — обучение финальной модели на полном наборе
- inspect_zero_shot_results.py — визуальная проверка zero-shot результатов
- inspect_trained_results.py — визуальная проверка результатов обученной модели


## Инференс 
На вход подаются
- изображение 
- классы МКТУ входного тз

Система: 
1. вычисляет embedding входного изображения
2. сравнивает его с датасетом 
3. находит 5 похожих знаков 
4. считает риск
5. сохраняет результаты

#### Инференс и риск
- inference_search.py — поиск похожих товарных знаков по новой картинке
- risk_utils.py — логика расчета риска

#### Эксперимент с hard negatives
- train_metric_neg.py — CV-обучение с hard negatives
- retrieval_dataset_neg.py — dataset для сценария с hard negatives
- build_hard_negatives_from_cv.py — сбор hard negative примеров по результатам CV
- prepare_manual_annotation.py — подготовка кандидатов для ручной разметки


## Установка зависимостей

```bash
pip install -r requirements.txt
```


## Общая логика пайплайна

Полный пайплайн выглядит так:

1. Подготовить изображения
2. Собрать метаданные изображений
3. Извлечь классы МКТУ из PDF
4. Объединить МКТУ с изображениями
5. Добавить split-информацию
6. Построить retrieval folds
7. Посчитать zero-shot качество
8. Обучить retrieval-модель по фолдам
9. Проанализировать результаты
10. Построить финальный retrieval-набор
11. Обучить финальную модель
12. Запустить инференс
13. При необходимости отдельно проверить hard negatives



# 1. Подготовка данных

## 1.1. Организация исходных изображений

Скрипт `organize_fips.py`:
- читает изображения из папки `logos`;
- формирует структуру `dataset/<tm_id>/registry`;
- копирует или перемещает файлы.

Запуск:
```bash
python organize_fips.py
```


## 1.3. Сканирование датасета

Скрипт `scan_dataset.py`:
- читает структуру папки `dataset/`;
- собирает информацию по всем изображениям;
- создаёт:
  - `prepared/images_metadata.csv`
  - `prepared/tm_metadata_template.csv`
  - `prepared/all_tm_ids.txt`

`scan_dataset.py` использует пути из `config.py`.

Запуск:
```bash
python scan_dataset.py
```


## 1.4. Заполнение МКТУ из PDF

Скрипт `fill_mktu_from_pdf.py`:
- читает PDF с классами МКТУ;
- дополняет шаблон метаданных;
- сохраняет:
  - `prepared/tm_metadata_filled.csv`

Запуск:
```bash
python fill_mktu_from_pdf.py
```




## 1.5. Объединение МКТУ с метаданными изображений

Скрипт `merge_mktu_into_images.py`:
- объединяет `prepared/images_metadata.csv`
- и `prepared/tm_metadata_filled.csv`

Результат:
- `prepared/images_metadata_with_mktu.csv`

Запуск:
```bash
python merge_mktu_into_images.py
```



## 1.6. Добавление split в итоговый CSV

Скрипт `split_to_images.py`:
- читает `prepared/images_metadata_with_mktu.csv`;
- читает:
  - `prepared/splits/train_ids.txt`
  - `prepared/splits/val_ids.txt`
  - `prepared/splits/test_ids.txt`;
- добавляет колонку `split`;
- сохраняет:
  - `prepared/images_metadata_final.csv`

Запуск:
```bash
python split_to_images.py
```



# 2. Построение retrieval folds

Скрипт `build_retrieval_folds.py` строит fold-ы для retrieval-задачи.

## Пример запуска
```bash
python build_retrieval_folds.py \
  --input prepared/images_metadata_final.csv \
  --out_dir prepared/retrieval_cv \
  --n_folds 5 \
  --seed 42
```

# 3. Zero-shot оценка retrieval-качества

Скрипт `evaluate_retrieval.py` считает retrieval-метрики без дообучения и сохраняет top-k предсказания по fold-ам.

## Пример запуска
```bash
python evaluate_retrieval.py \
  --cv_dir prepared/retrieval_cv \
  --project_root . \
  --model_name resnet50 \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 0 \
  --device auto
```


## Сохраняет:

В каждом `fold_*`:
- `zero_shot_metrics.csv`
- `zero_shot_top5_predictions.csv`

В корне `cv_dir`:
- `zero_shot_cv_metrics.csv`
- `zero_shot_summary.json`



# 4. Основное CV-обучение: `train_metric.py`

Это **основной рабочий сценарий** обучения retrieval-модели по фолдам.


## Пример запуска
```bash
python train_metric.py \
  --cv_dir prepared/retrieval_cv \
  --project_root . \
  --output_dir checkpoints_metric \
  --model_name resnet50 \
  --image_size 224 \
  --emb_dim 256 \
  --epochs 12 \
  --batch_size 16 \
  --num_workers 0 \
  --lr_backbone 3e-6 \
  --lr_head 3e-4 \
  --weight_decay 1e-4 \
  --temperature 0.07 \
  --device auto
```


## Сохраняет

Для каждого `fold_*`:
- `best_model.pt`
- `best_val_top5_predictions.csv`
- `history.csv`
- `best_metrics.json`

В корне `output_dir`:
- `cv_best_metrics.csv`
- `cv_summary.json`


# 5. Визуальная проверка результатов

## 5.1. Zero-shot результаты

Скрипт `inspect_zero_shot_results.py`:
- читает `zero_shot_top5_predictions.csv`;
- собирает визуальные коллажи query / top-k.

  Запуск:
```bash
python inspect_zero_shot_results.py
```


## 5.2. Результаты обученной модели

Скрипт `inspect_trained_results.py`:
- читает `best_val_top5_predictions.csv`;
- собирает визуальные таблицы по fold-ам.

Запуск:
```bash
python inspect_trained_results.py
```


# 6. Построение финального retrieval-набора

После выбора рабочей конфигурации можно собрать полный retrieval-набор для финального обучения.

## Пример запуска
```bash
python build_final_retrieval_set.py \
  --input prepared/images_metadata_final.csv \
  --out_dir prepared/final_retrieval
```


# 7. Обучение финальной модели: `train_final_model.py`

Этот скрипт обучает итоговую retrieval-модель на полном наборе.

## Пример запуска
```bash
python train_final_model.py \
  --final_dir prepared/final_retrieval \
  --project_root . \
  --output_dir final_model \
  --model_name resnet50 \
  --image_size 224 \
  --emb_dim 256 \
  --epochs 10 \
  --batch_size 16 \
  --num_workers 0 \
  --lr_backbone 3e-6 \
  --lr_head 3e-4 \
  --weight_decay 1e-4 \
  --temperature 0.07 \
  --device auto
```



## Сохраняет

В `output_dir`:
- `best_model.pt`
- `last_model.pt`
- `best_final_top5_predictions.csv`
- `gallery_metadata.csv`
- `gallery_embeddings.pt`
- `best_sanity_metrics.json`
- `history.csv`
- `final_model_summary.json`

# 8. Инференс: поиск похожих знаков по новой картинке

Скрипт `inference_search.py` принимает:
- входную картинку;
- папку с финальной моделью;
- `top_k`;
- классы МКТУ для нового знака.

## Пример запуска
```bash
python inference_search.py \
  --image_path path/to/query.jpg \
  --model_dir final_model \
  --project_root . \
  --output_dir inference_outputs \
  --top_k 5 \
  --query_mktu_classes "25 35" \
  --device auto
```



## Сохраняет

В `output_dir`:
- `<image_stem>_top{k}.csv`
- `<image_stem>_top{k}.json`
- `<image_stem>_risk.json`
- `<image_stem>_top{k}.jpg`

Скрипт также печатает:
- top-k найденные знаки;
- `risk_score`;
- `risk_level`.



## Минимальный набор команд

```bash
cd ml_project
pip install -r requirements.txt
python scan_dataset.py
python fill_mktu_from_pdf.py
python merge_mktu_into_images.py
python split_to_images.py
python build_retrieval_folds.py
python train_metric.py
python build_final_retrieval_set.py
python train_final_model.py
python inference_search.py --image_path path/to/query.jpg --query_mktu_classes "25 35"
```
