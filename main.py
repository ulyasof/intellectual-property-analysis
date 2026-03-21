from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from typing import Any, Dict, List, Optional

from data_loader import (
    filter_companies,
    get_companies,
    get_available_columns,
    get_brand_score_distribution,
    get_cluster_stats,
    get_companies_short,
    get_company_by_inn,
    get_numeric_summary,
    get_pca_data,
    get_top_companies_by_brand_score,
)
from model_adapter import (
    InvalidImageError,
    ModelExecutionError,
    ModelResultError,
    find_similar_logos,
    get_model_info,
)

app = FastAPI(
    title="Trademark Analytics Service",
    description="Микросервис для аналитики компаний и поиска похожих логотипов",
    version="1.0.0",
)


ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


@app.get("/")
def root():
    return {
        "message": "Trademark Analytics Service is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "trademark-analytics-service",
    }


@app.get("/columns")
def columns():
    """
    Возвращает список колонок, доступных в итоговом датасете.
    """
    try:
        return {"columns": get_available_columns()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении колонок: {e}")


@app.get("/summary")
def summary():
    """
    Возвращает краткую сводку по датасету:
    - число строк
    - список колонок
    - числовые колонки
    - простую статистику
    """
    try:
        return get_numeric_summary()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при построении summary: {e}")


@app.get("/companies")
def companies(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        columns: Optional[List[str]] = Query(
            None,
            description="Выбранные колонки. Можно передавать несколько раз: ?columns=inn&columns=company_name",
        ),
):
    """
    Возвращает часть компаний из итогового датасета.

    Примеры:
    /companies
    /companies?limit=20&offset=40
    /companies?columns=inn&columns=company_name
    /companies?limit=10&columns=inn&columns=company_name&columns=brand_score
    """
    try:
        return get_companies(
            limit=limit,
            offset=offset,
            columns=columns,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении компаний: {e}")


@app.get("/companies_short")
def companies_short(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Возвращает сокращённый список компаний.
    """
    try:
        return get_companies_short(limit=limit, offset=offset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка компаний: {e}")

@app.get("/company/{inn}")
def company_by_inn(inn: str):
    """
    Возвращает данные одной компании по ИНН.
    """
    try:
        company = get_company_by_inn(inn)

        if company is None:
            raise HTTPException(status_code=404, detail="Компания не найдена")

        return company
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске компании: {e}")


@app.get("/clusters")
def clusters():
    """
    Возвращает количество компаний в каждом кластере.
    Подходит для bar chart.
    """
    try:
        return get_cluster_stats()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении кластеров: {e}")


@app.get("/brand_score")
def brand_score():
    """
    Возвращает данные по индексу силы бренда.
    """
    try:
        return get_brand_score_distribution()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении brand_score: {e}")


@app.get("/brand_score/top")
def top_brand_score(
    limit: int = Query(10, ge=1, le=100),
):
    """
    Возвращает top-N компаний по индексу силы бренда.
    """
    try:
        return get_top_companies_by_brand_score(limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении top brand_score: {e}")


@app.get("/pca")
def pca():
    """
    Возвращает точки для PCA/scatter plot.
    """
    try:
        return get_pca_data()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении PCA-данных: {e}")


@app.get("/filter")
def companies_filter(
    cluster: Optional[int] = Query(None),
    min_brand_score: Optional[float] = Query(None, ge=0.0),
    max_brand_score: Optional[float] = Query(None, ge=0.0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    columns: Optional[List[str]] = Query(
        None,
        description="Выбранные колонки. Можно передавать несколько раз: ?columns=inn&columns=company_name",
    ),
):
    """
    Простая фильтрация компаний.

    Примеры:
    /filter?cluster=2
    /filter?min_brand_score=0.5
    /filter?cluster=1&min_brand_score=0.4&max_brand_score=0.9
    /filter?cluster=1&limit=20&offset=40
    /filter?cluster=1&columns=inn&columns=company_name&columns=brand_score
    """
    try:
        if (
                min_brand_score is not None
                and max_brand_score is not None
                and min_brand_score > max_brand_score
        ):
            raise HTTPException(
                status_code=400,
                detail="min_brand_score не может быть больше max_brand_score",
            )

        return filter_companies(
            cluster=cluster,
            min_brand_score=min_brand_score,
            max_brand_score=max_brand_score,
            limit=limit,
            offset=offset,
            columns=columns,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при фильтрации компаний: {e}")


@app.get("/model_info")
def model_info():
    """
    Возвращает информацию о подключённой модели поиска логотипов.
    """
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении информации о модели: {e}")


@app.post("/search_logo")
async def search_logo(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Принимает изображение логотипа и возвращает список похожих результатов.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат файла. Разрешены PNG, JPEG, JPG, WEBP.",
        )

    try:
        file_bytes = await file.read()

        results = find_similar_logos(
            file_bytes=file_bytes,
            top_k=top_k,
        )

        response = {
            "filename": file.filename,
            "content_type": file.content_type,
            "top_k": top_k,
            "results_count": len(results),
            "results": results,
        }

        if not results:
            response["message"] = "Похожие логотипы не найдены"

        return response

    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelExecutionError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка выполнения модели: {e}",
        )
    except ModelResultError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки результатов модели: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Неожиданная ошибка при поиске похожих логотипов: {e}",
        )