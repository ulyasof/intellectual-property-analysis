from typing import Any, Dict, List, Optional
import pandas as pd

MOCK_COMPANIES: List[Dict[str, Any]] = [
    {
        "inn": "7701234567",
        "company_name": "ООО Ромашка",
        "region": "Москва",
        "industry": "Ритейл",
        "num_marks": 12,
        "cluster": 1,
        "brand_score": 0.81,
        "active_share": 0.72,
        "avg_portfolio_age": 5.4,
        "nice_class_count": 6,
        "pca_x": 1.2,
        "pca_y": 0.8,
    },
    {
        "inn": "7801234567",
        "company_name": "ООО Василек",
        "region": "Санкт-Петербург",
        "industry": "IT",
        "num_marks": 7,
        "cluster": 2,
        "brand_score": 0.63,
        "active_share": 0.66,
        "avg_portfolio_age": 3.8,
        "nice_class_count": 4,
        "pca_x": -0.7,
        "pca_y": 1.0,
    },
    {
        "inn": "1651234567",
        "company_name": "ООО Лотос",
        "region": "Казань",
        "industry": "Производство",
        "num_marks": 20,
        "cluster": 1,
        "brand_score": 0.88,
        "active_share": 0.79,
        "avg_portfolio_age": 6.2,
        "nice_class_count": 8,
        "pca_x": 1.8,
        "pca_y": 0.2,
    },
    {
        "inn": "5401234567",
        "company_name": "ООО Сибирь Бренд",
        "region": "Новосибирск",
        "industry": "Логистика",
        "num_marks": 5,
        "cluster": 3,
        "brand_score": 0.42,
        "active_share": 0.58,
        "avg_portfolio_age": 2.9,
        "nice_class_count": 3,
        "pca_x": -1.4,
        "pca_y": -0.8,
    },
    {
        "inn": "6601234567",
        "company_name": "ООО УралМарка",
        "region": "Екатеринбург",
        "industry": "Промышленность",
        "num_marks": 15,
        "cluster": 2,
        "brand_score": 0.71,
        "active_share": 0.68,
        "avg_portfolio_age": 5.1,
        "nice_class_count": 7,
        "pca_x": -0.3,
        "pca_y": 1.4,
    },
    {
        "inn": "2301234567",
        "company_name": "ООО Юг Торг",
        "region": "Краснодар",
        "industry": "Торговля",
        "num_marks": 9,
        "cluster": 1,
        "brand_score": 0.69,
        "active_share": 0.73,
        "avg_portfolio_age": 4.0,
        "nice_class_count": 5,
        "pca_x": 1.0,
        "pca_y": -0.1,
    },
    {
        "inn": "5001234567",
        "company_name": "ООО Подмосковье Лого",
        "region": "Московская область",
        "industry": "Услуги",
        "num_marks": 11,
        "cluster": 2,
        "brand_score": 0.66,
        "active_share": 0.65,
        "avg_portfolio_age": 4.8,
        "nice_class_count": 5,
        "pca_x": -0.1,
        "pca_y": 0.9,
    },
    {
        "inn": "6101234567",
        "company_name": "ООО Дон Бренд",
        "region": "Ростов-на-Дону",
        "industry": "Агро",
        "num_marks": 4,
        "cluster": 3,
        "brand_score": 0.39,
        "active_share": 0.57,
        "avg_portfolio_age": 2.4,
        "nice_class_count": 2,
        "pca_x": -1.2,
        "pca_y": -1.1,
    },
    {
        "inn": "0201234567",
        "company_name": "ООО БашМаркет",
        "region": "Уфа",
        "industry": "Торговля",
        "num_marks": 13,
        "cluster": 1,
        "brand_score": 0.76,
        "active_share": 0.75,
        "avg_portfolio_age": 5.7,
        "nice_class_count": 6,
        "pca_x": 1.5,
        "pca_y": -0.4,
    },
    {
        "inn": "5901234567",
        "company_name": "ООО Пермь Знак",
        "region": "Пермь",
        "industry": "Производство",
        "num_marks": 6,
        "cluster": 2,
        "brand_score": 0.58,
        "active_share": 0.61,
        "avg_portfolio_age": 3.3,
        "nice_class_count": 4,
        "pca_x": -0.5,
        "pca_y": 0.6,
    },
    {
        "inn": "7401234567",
        "company_name": "ООО ЧелБренд",
        "region": "Челябинск",
        "industry": "Металлургия",
        "num_marks": 10,
        "cluster": 3,
        "brand_score": 0.47,
        "active_share": 0.62,
        "avg_portfolio_age": 4.1,
        "nice_class_count": 5,
        "pca_x": -1.0,
        "pca_y": -0.5,
    },
    {
        "inn": "3601234567",
        "company_name": "ООО Воронеж Марка",
        "region": "Воронеж",
        "industry": "Пищевая промышленность",
        "num_marks": 8,
        "cluster": 1,
        "brand_score": 0.64,
        "active_share": 0.70,
        "avg_portfolio_age": 4.6,
        "nice_class_count": 4,
        "pca_x": 0.9,
        "pca_y": 0.4,
    },
]


def load_dataset() -> pd.DataFrame:
    return pd.DataFrame(MOCK_COMPANIES)


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {key: _normalize_value(value) for key, value in record.items()}

    if "trademark_count" in normalized and "num_marks" not in normalized:
        normalized["num_marks"] = normalized["trademark_count"]

    if "num_marks" in normalized and "trademark_count" not in normalized:
        normalized["trademark_count"] = normalized["num_marks"]

    if "avg_portfolio_age" in normalized and "avg_age" not in normalized:
        normalized["avg_age"] = normalized["avg_portfolio_age"]

    if "avg_age" in normalized and "avg_portfolio_age" not in normalized:
        normalized["avg_portfolio_age"] = normalized["avg_age"]

    return normalized

def get_companies(
    limit: int = 100,
    offset: int = 0,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    df = load_dataset()

    if columns:
        safe_columns = [col for col in columns if col in df.columns]
        if safe_columns:
            df = df[safe_columns]

    total = len(df)
    page_df = df.iloc[offset: offset + limit].copy()

    items = [_normalize_record(row) for row in page_df.to_dict(orient="records")]

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def get_company_by_inn(inn: str) -> Optional[Dict[str, Any]]:
    df = load_dataset()
    row = df[df["inn"] == inn]

    if row.empty:
        return None

    record = row.iloc[0].to_dict()
    return _normalize_record(record)


def get_companies_short(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    df = load_dataset()

    preferred_columns = [
        "inn",
        "company_name",
        "region",
        "industry",
        "num_marks",
        "cluster",
    ]
    safe_columns = [col for col in preferred_columns if col in df.columns]

    total = len(df)
    page_df = df.iloc[offset: offset + limit][safe_columns].copy()
    items = [_normalize_record(row) for row in page_df.to_dict(orient="records")]

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def get_cluster_stats() -> List[Dict[str, Any]]:
    df = load_dataset()

    cluster_counts = (
        df.groupby("cluster")
        .size()
        .reset_index(name="count")
        .sort_values("cluster")
    )

    return [
        _normalize_record(row)
        for row in cluster_counts.to_dict(orient="records")
    ]


def get_brand_score_distribution() -> List[Dict[str, Any]]:
    df = load_dataset()

    columns = ["brand_score"]
    if "company_name" in df.columns:
        columns.append("company_name")
    if "inn" in df.columns:
        columns.append("inn")

    data = df[columns].sort_values("brand_score", ascending=False)
    return [_normalize_record(row) for row in data.to_dict(orient="records")]


def get_top_companies_by_brand_score(limit: int = 10) -> List[Dict[str, Any]]:
    df = load_dataset()

    columns = ["company_name", "inn", "brand_score"]
    safe_columns = [col for col in columns if col in df.columns]

    data = df[safe_columns].sort_values("brand_score", ascending=False).head(limit)
    return [_normalize_record(row) for row in data.to_dict(orient="records")]


def get_pca_data() -> List[Dict[str, Any]]:
    df = load_dataset()

    columns = ["pca_x", "pca_y"]
    for optional_col in ["cluster", "company_name", "inn", "brand_score", "num_marks"]:
        if optional_col in df.columns:
            columns.append(optional_col)

    data = df[columns].copy()
    return [_normalize_record(row) for row in data.to_dict(orient="records")]


def get_numeric_summary() -> Dict[str, Any]:
    df = load_dataset()

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    summary = {}
    for col in numeric_columns:
        summary[col] = {
            "min": _normalize_value(df[col].min()),
            "max": _normalize_value(df[col].max()),
            "mean": _normalize_value(round(float(df[col].mean()), 3)),
        }

    return {
        "row_count": int(len(df)),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "summary": summary,
    }


def get_available_columns() -> List[str]:
    df = load_dataset()
    return df.columns.tolist()


def filter_companies(
    cluster: Optional[int] = None,
    min_brand_score: Optional[float] = None,
    max_brand_score: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    df = load_dataset()

    if cluster is not None:
        df = df[df["cluster"] == cluster]

    if min_brand_score is not None:
        df = df[df["brand_score"] >= min_brand_score]

    if max_brand_score is not None:
        df = df[df["brand_score"] <= max_brand_score]

    total = len(df)
    page_df = df.iloc[offset: offset + limit].copy()

    items = [_normalize_record(row) for row in page_df.to_dict(orient="records")]

    return {
        "items": items,
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


from typing import Optional, Dict, Any, List
import os

def search_similar_logos(file_path: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
    query_image_name = os.path.basename(file_path) if file_path else None

    mock_matches: List[Dict[str, Any]] = [
        {
            "trademark_id": "TM-1001",
            "name": "ROMASHKA",
            "similarity": 0.94,
            "image_url": None,
            "candidate_mktu_classes": [3, 5, 35],
            "comment": "Очень высокое визуальное сходство.",
        },
        {
            "trademark_id": "TM-1048",
            "name": "ROMA",
            "similarity": 0.88,
            "image_url": None,
            "candidate_mktu_classes": [3, 35],
            "comment": "Высокое сходство по композиции и написанию.",
        },
        {
            "trademark_id": "TM-2050",
            "name": "ROSA",
            "similarity": 0.79,
            "image_url": None,
            "candidate_mktu_classes": [5, 44],
            "comment": "Есть сходство по форме и визуальной подаче.",
        },
        {
            "trademark_id": "TM-3331",
            "name": "ROMASH",
            "similarity": 0.73,
            "image_url": None,
            "candidate_mktu_classes": [35],
            "comment": "Умеренное сходство, риск средний.",
        },
        {
            "trademark_id": "TM-4102",
            "name": "FLOWERMARK",
            "similarity": 0.61,
            "image_url": None,
            "candidate_mktu_classes": [3, 30],
            "comment": "Сходство ограничено отдельными элементами.",
        },
    ]

    results = mock_matches[:top_k]

    query_mktu_classes = [3, 35]

    for item in results:
        overlap = sorted(set(query_mktu_classes) & set(item["candidate_mktu_classes"]))
        item["mktu_overlap"] = overlap

    top_match = results[0] if results else None
    max_similarity = top_match["similarity"] if top_match else 0.0
    top_overlap = top_match["mktu_overlap"] if top_match else []

    if max_similarity >= 0.9 and top_overlap:
        risk_level = "high"
        explanation = "Найдено очень сильное визуальное сходство и есть пересечение по классам МКТУ."
    elif max_similarity >= 0.75:
        risk_level = "medium"
        explanation = "Найдено заметное визуальное сходство, требуется дополнительная проверка."
    else:
        risk_level = "low"
        explanation = "Сильных конфликтов по моковым данным не обнаружено."

    risk = {
        "risk_level": risk_level,
        "risk_score": round(max_similarity, 3),
        "top_conflict_tm_id": top_match["trademark_id"] if top_match else None,
        "top_visual_score": round(max_similarity, 3),
        "num_unique_candidates": len(results),
        "query_mktu_classes": query_mktu_classes,
        "candidate_mktu_classes": top_match["candidate_mktu_classes"] if top_match else [],
        "mktu_overlap": top_overlap,
        "explanation": explanation,
    }

    return {
        "query_image_name": query_image_name,
        "results": {
            "results": results,
            "risk": risk,
        },
    }