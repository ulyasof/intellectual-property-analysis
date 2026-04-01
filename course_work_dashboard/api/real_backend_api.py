from typing import Any, Dict, List, Optional
import os
import requests
import mimetypes


#BASE_URL = "http://127.0.0.1:8000" # для запуска на сервере
BASE_URL = os.getenv("BACKEND_URL", "http://194.67.102.116:8000")  # для локального запуска
API_TOKEN = os.getenv("API_TOKEN")


def _headers() -> Dict[str, str]:
    headers = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def get_companies(
    limit: int = 100,
    offset: int = 0,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    params = {"limit": limit, "offset": offset}
    if columns:
        params["columns"] = columns

    response = requests.get(
        f"{BASE_URL}/companies",
        params=params,
        headers=_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def get_companies_short(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/companies_short",
        params={"limit": limit, "offset": offset},
        headers=_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def get_company_by_inn(inn: str) -> Optional[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_URL}/company/{inn}",
        headers=_headers(),
        timeout=15,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def get_cluster_stats() -> List[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_URL}/clusters",
        headers=_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()

def get_company_trademark_classes(inn: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/company/{inn}/trademarks",
        headers=_headers(),
        timeout=15,
    )
    if response.status_code == 404:
        return {"inn": inn, "count": 0, "total": 0, "limit": 100, "offset": 0, "items": []}
    response.raise_for_status()
    return response.json()


def get_company_trademark_classes_agg(inn: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/company/{inn}/mktu_stats",
        headers=_headers(),
        timeout=15,
    )
    if response.status_code == 404:
        return {"inn": inn, "unique_classes": [], "classes_count": {}}
    response.raise_for_status()
    return response.json()


def get_brand_score_distribution() -> List[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_URL}/brand_score",
        headers=_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def get_top_companies_by_brand_score(limit: int = 10) -> List[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_URL}/brand_score/top",
        params={"limit": limit},
        headers=_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def get_pca_data() -> List[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_URL}/pca",
        headers=_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def get_numeric_summary() -> Dict[str, Any]:
    response = requests.get(f"{BASE_URL}/summary", headers=_headers(), timeout=15)
    response.raise_for_status()
    raw = response.json()

    return {
        "row_count": raw.get("rows_count"),
        "columns": raw.get("columns", []),
        "numeric_columns": raw.get("numeric_columns", []),
        "stats": raw.get("stats", {}),
    }


def get_available_columns():
    response = requests.get(f"{BASE_URL}/columns", headers=_headers(), timeout=15)
    response.raise_for_status()
    raw = response.json()

    if isinstance(raw, dict) and "columns" in raw:
        return raw["columns"]
    return raw


def filter_companies(
    cluster: Optional[int] = None,
    min_brand_score: Optional[float] = None,
    max_brand_score: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    params = {"limit": limit, "offset": offset}
    if cluster is not None:
        params["cluster"] = cluster
    if min_brand_score is not None:
        params["min_brand_score"] = min_brand_score
    if max_brand_score is not None:
        params["max_brand_score"] = max_brand_score

    response = requests.get(
        f"{BASE_URL}/filter",
        params=params,
        headers=_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()

def search_similar_logos(
    file_path: Optional[str] = None,
    top_k: int = 5,
    query_mktu_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if not file_path:
        raise ValueError("Не указан путь к файлу")

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(file_path, "rb") as f:
        files = {
            "file": (os.path.basename(file_path), f, mime_type)
        }

        data = {"top_k": top_k}
        if query_mktu_classes:
            data["query_mktu"] = ",".join(str(x) for x in query_mktu_classes)

        response = requests.post(
            f"{BASE_URL}/search_logo",
            headers=_headers(),
            files=files,
            data=data,
            timeout=60,
        )

    if not response.ok:
        raise RuntimeError(f"Ошибка {response.status_code}: {response.text}")

    return response.json()