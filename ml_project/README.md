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

## Основные файлы 

- inference_search.py - по картинке ищет похожие товарные знаки 
- risk_utils.py - логика расчета риска
- retrieval_dataset.py - общий модуль данных для обучения модели
- train_metric.py - обучение модели по фолдам
- train_final_model.py - обучение финальной модели
- evaluate_retrieval.py - оценка качества 
- config.py - настройки и пути
- utils.py - вспомогательные функции

## Подготовка данных 

- organize_fips.py
- rename_real.py
- scan_dataset.py
- fill_mktu_from_pdf.py
- merge_mktu_into_images.py
- split_to_images.py
- build_retrieval_folds.py
- build_final_retrieval_set.py


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

## Пример запуска

```bash
python3 inference_search.py \
  --image_path path/to/query.jpg \
  --model_dir final_model \
  --project_root . \
  --output_dir inference_outputs \
  --top_k 5 \
  --query_mktu_classes "25 35"