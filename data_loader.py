from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DATA_PATH = Path("companies_dataset.csv")


def load_dataset() -> pd.DataFrame:
    """
    Загружает итоговый датасет компаний из CSV.

    Важно:
    - inn читаем как строку, чтобы не ломались ведущие нули и формат;
    - если файла нет, выбрасываем понятную ошибку.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Файл датасета не найден: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH, dtype={"inn": str})
    return df


def _normalize_value(value: Any) -> Any:
    """
    Преобразует NaN/NaT в None, чтобы JSON-ответы были аккуратными.
    """
    if pd.isna(value):
        return None
    return value


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводит одну запись к виду, удобному для JSON: NaN -> None
    """
    return {key: _normalize_value(value) for key, value in record.items()}


def get_companies(limit: int = 100, offset: int = 0, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Возвращает часть компаний из датасета.

    Параметры:
    - limit: сколько записей вернуть
    - offset: с какой записи начать
    - columns: какие колонки вернуть; если None, возвращаются все
    """
    df = load_dataset()

    total = len(df)

    if columns:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"В датасете отсутствуют колонки: {', '.join(missing_columns)}"
            )
        df = df[columns]

    result = df.iloc[offset: offset + limit]

    records = result.to_dict(orient="records")
    items = [_normalize_record(record) for record in records]

    return {
        "limit": limit,
        "offset": offset,
        "columns": columns,
        "count": len(items),
        "total": total,
        "items": items,
    }


def get_company_by_inn(inn: str) -> Optional[Dict[str, Any]]:
    """
    Ищет компанию по ИНН.
    Если компания не найдена, возвращает None.
    """
    df = load_dataset()

    if "inn" not in df.columns:
        raise ValueError("В датасете отсутствует колонка 'inn'")

    row = df[df["inn"] == inn]

    if row.empty:
        return None

    record = row.iloc[0].to_dict()
    return _normalize_record(record)


def get_companies_short(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """
    Возвращает короткий список компаний.
    Подходит для таблицы или выпадающего списка на дашборде.

    Пытается отдать только самые важные колонки, если они существуют.
    """
    df = load_dataset()

    preferred_columns = [
        "inn",
        "company_name",
        "brand_score",
        "cluster",
        "revenue",
        "trademarks_count",
    ]

    existing_columns = [col for col in preferred_columns if col in df.columns]
    total = len(df)

    if existing_columns:
        df = df[existing_columns]

    result = df.iloc[offset: offset + limit]

    records = result.to_dict(orient="records")
    items = [_normalize_record(record) for record in records]

    return {
        "limit": limit,
        "offset": offset,
        "columns": existing_columns if existing_columns else list(df.columns),
        "count": len(items),
        "total": total,
        "items": items,
    }


def get_cluster_stats() -> List[Dict[str, Any]]:
    """
    Возвращает количество компаний в каждом кластере.

    Формат:
    [
        {"cluster": 0, "count": 10},
        {"cluster": 1, "count": 25},
        ...
    ]
    """
    df = load_dataset()

    if "cluster" not in df.columns:
        raise ValueError("В датасете отсутствует колонка 'cluster'")

    stats = (
        df.groupby("cluster")
        .size()
        .reset_index(name="count")
        .sort_values("cluster")
    )

    records = stats.to_dict(orient="records")
    return [_normalize_record(record) for record in records]


def get_brand_score_distribution() -> List[Dict[str, Any]]:
    """
    Возвращает данные для графика или таблицы по индексу силы бренда.

    Если есть company_name и inn, они тоже добавляются.
    """
    df = load_dataset()

    if "brand_score" not in df.columns:
        raise ValueError("В датасете отсутствует колонка 'brand_score'")

    columns = ["brand_score"]

    if "inn" in df.columns:
        columns.append("inn")
    if "company_name" in df.columns:
        columns.append("company_name")
    if "cluster" in df.columns:
        columns.append("cluster")

    result = df[columns].sort_values("brand_score", ascending=False)

    records = result.to_dict(orient="records")
    return [_normalize_record(record) for record in records]


def get_top_companies_by_brand_score(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Возвращает top-N компаний по индексу бренда.
    """
    df = load_dataset()

    if "brand_score" not in df.columns:
        raise ValueError("В датасете отсутствует колонка 'brand_score'")

    columns = ["brand_score"]

    if "inn" in df.columns:
        columns.append("inn")
    if "company_name" in df.columns:
        columns.append("company_name")
    if "cluster" in df.columns:
        columns.append("cluster")
    if "revenue" in df.columns:
        columns.append("revenue")
    if "trademarks_count" in df.columns:
        columns.append("trademarks_count")

    result = (
        df[columns]
        .sort_values("brand_score", ascending=False)
        .head(limit)
    )

    records = result.to_dict(orient="records")
    return [_normalize_record(record) for record in records]


def get_pca_data() -> List[Dict[str, Any]]:
    """
    Возвращает данные для PCA/scatter plot.

    Дополнительно, если есть:
    - cluster
    - company_name
    - inn
    - brand_score
    """
    df = load_dataset()

    required_columns = ["pca_x", "pca_y"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"В датасете отсутствует колонка '{col}'")

    columns = ["pca_x", "pca_y"]

    optional_columns = ["cluster", "company_name", "inn", "brand_score"]
    for col in optional_columns:
        if col in df.columns:
            columns.append(col)

    result = df[columns]

    records = result.to_dict(orient="records")
    return [_normalize_record(record) for record in records]


def get_numeric_summary() -> Dict[str, Any]:
    """
    Возвращает простую сводку по числовым колонкам:
    - количество строк
    - список колонок
    - min/max/mean для некоторых числовых признаков

    Это удобно для отладки или общего endpoint /summary.
    """
    df = load_dataset()

    summary: Dict[str, Any] = {
        "rows_count": int(len(df)),
        "columns": list(df.columns),
        "numeric_columns": [],
        "stats": {},
    }

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    summary["numeric_columns"] = numeric_columns

    for col in numeric_columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        summary["stats"][col] = {
            "min": _normalize_value(series.min()),
            "max": _normalize_value(series.max()),
            "mean": _normalize_value(series.mean()),
        }

    return summary


def get_available_columns() -> List[str]:
    """
    Возвращает список всех колонок датасета.
    """
    df = load_dataset()
    return list(df.columns)


def filter_companies(
    cluster: Optional[int] = None,
    min_brand_score: Optional[float] = None,
    max_brand_score: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Простая фильтрация компаний.

    Параметры:
    - cluster: оставить только компании выбранного кластера
    - min_brand_score: нижняя граница brand_score
    - max_brand_score: верхняя граница brand_score
    - limit: сколько записей вернуть
    - offset: с какой записи начать
    - columns: какие колонки вернуть; если None, возвращаются все
    """
    df = load_dataset()

    result = df.copy()

    if cluster is not None:
        if "cluster" not in result.columns:
            raise ValueError("В датасете отсутствует колонка 'cluster'")
        result = result[result["cluster"] == cluster]

    if min_brand_score is not None:
        if "brand_score" not in result.columns:
            raise ValueError("В датасете отсутствует колонка 'brand_score'")
        result = result[result["brand_score"] >= min_brand_score]

    if max_brand_score is not None:
        if "brand_score" not in result.columns:
            raise ValueError("В датасете отсутствует колонка 'brand_score'")
        result = result[result["brand_score"] <= max_brand_score]

    total = len(result)

    if columns:
        missing_columns = [col for col in columns if col not in result.columns]
        if missing_columns:
            raise ValueError(
                f"В датасете отсутствуют колонки: {', '.join(missing_columns)}"
            )
        result = result[columns]

    paged_result = result.iloc[offset: offset + limit]

    records = paged_result.to_dict(orient="records")
    items = [_normalize_record(record) for record in records]

    return {
        "filters": {
            "cluster": cluster,
            "min_brand_score": min_brand_score,
            "max_brand_score": max_brand_score,
        },
        "limit": limit,
        "offset": offset,
        "columns": columns,
        "count": len(items),
        "total": total,
        "items": items,
    }