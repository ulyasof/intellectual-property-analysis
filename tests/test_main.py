from fastapi.testclient import TestClient
import main


client = TestClient(main.app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Trademark Analytics Service is running",
        "docs": "/docs",
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "trademark-analytics-service"


def test_columns_success(monkeypatch):
    monkeypatch.setattr(main, "get_available_columns", lambda: ["inn", "company_name"])

    response = client.get("/columns")

    assert response.status_code == 200
    assert response.json() == {"columns": ["inn", "company_name"]}


def test_columns_file_not_found(monkeypatch):
    monkeypatch.setattr(main, "get_available_columns", lambda: (_ for _ in ()).throw(FileNotFoundError("file missing")))

    response = client.get("/columns")

    assert response.status_code == 500
    assert response.json()["detail"] == "file missing"


def test_summary_success(monkeypatch):
    monkeypatch.setattr(main, "get_numeric_summary", lambda: {"rows_count": 3, "columns": ["inn"]})

    response = client.get("/summary")

    assert response.status_code == 200
    assert response.json()["rows_count"] == 3


def test_companies_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_companies",
        lambda limit, offset, columns: {
            "limit": limit,
            "offset": offset,
            "columns": columns,
            "count": 1,
            "total": 1,
            "items": [{"inn": "123"}],
        },
    )

    response = client.get("/companies?limit=10&offset=0&columns=inn&columns=company_name")

    assert response.status_code == 200
    data = response.json()
    assert data["limit"] == 10
    assert data["columns"] == ["inn", "company_name"]
    assert data["items"][0]["inn"] == "123"


def test_companies_value_error(monkeypatch):
    monkeypatch.setattr(main, "get_companies", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad columns")))

    response = client.get("/companies")

    assert response.status_code == 400
    assert response.json()["detail"] == "bad columns"


def test_companies_short_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_companies_short",
        lambda limit, offset: {
            "limit": limit,
            "offset": offset,
            "count": 1,
            "total": 1,
            "items": [{"inn": "123", "company_name": "Alpha"}],
        },
    )

    response = client.get("/companies_short?limit=5&offset=1")

    assert response.status_code == 200
    assert response.json()["limit"] == 5
    assert response.json()["offset"] == 1


def test_company_by_inn_found(monkeypatch):
    monkeypatch.setattr(main, "get_company_by_inn", lambda inn: {"inn": inn, "company_name": "Alpha"})

    response = client.get("/company/123")

    assert response.status_code == 200
    assert response.json()["inn"] == "123"


def test_company_by_inn_not_found(monkeypatch):
    monkeypatch.setattr(main, "get_company_by_inn", lambda inn: None)

    response = client.get("/company/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Компания не найдена"


def test_company_by_inn_value_error(monkeypatch):
    monkeypatch.setattr(main, "get_company_by_inn", lambda inn: (_ for _ in ()).throw(ValueError("bad inn")))

    response = client.get("/company/123")

    assert response.status_code == 400
    assert response.json()["detail"] == "bad inn"


def test_clusters_success(monkeypatch):
    monkeypatch.setattr(main, "get_cluster_stats", lambda: [{"cluster": 0, "count": 2}])

    response = client.get("/clusters")

    assert response.status_code == 200
    assert response.json() == [{"cluster": 0, "count": 2}]


def test_brand_score_success(monkeypatch):
    monkeypatch.setattr(main, "get_brand_score_distribution", lambda: [{"brand_score": 0.9, "inn": "123"}])

    response = client.get("/brand_score")

    assert response.status_code == 200
    assert response.json()[0]["brand_score"] == 0.9


def test_top_brand_score_success(monkeypatch):
    monkeypatch.setattr(main, "get_top_companies_by_brand_score", lambda limit: [{"inn": "123", "brand_score": 0.9}])

    response = client.get("/brand_score/top?limit=3")

    assert response.status_code == 200
    assert response.json()[0]["inn"] == "123"


def test_pca_success(monkeypatch):
    monkeypatch.setattr(main, "get_pca_data", lambda: [{"pca_x": 1.0, "pca_y": 2.0}])

    response = client.get("/pca")

    assert response.status_code == 200
    assert response.json()[0]["pca_x"] == 1.0


def test_filter_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "filter_companies",
        lambda cluster, min_brand_score, max_brand_score, limit, offset, columns: {
            "filters": {
                "cluster": cluster,
                "min_brand_score": min_brand_score,
                "max_brand_score": max_brand_score,
            },
            "limit": limit,
            "offset": offset,
            "columns": columns,
            "count": 1,
            "total": 1,
            "items": [{"inn": "123", "brand_score": 0.8}],
        },
    )

    response = client.get("/filter?cluster=1&min_brand_score=0.5&max_brand_score=0.9&columns=inn")

    assert response.status_code == 200
    data = response.json()
    assert data["filters"]["cluster"] == 1
    assert data["filters"]["min_brand_score"] == 0.5
    assert data["filters"]["max_brand_score"] == 0.9


def test_filter_invalid_range():
    response = client.get("/filter?min_brand_score=1.0&max_brand_score=0.5")

    assert response.status_code == 400
    assert "min_brand_score не может быть больше max_brand_score" in response.json()["detail"]


def test_filter_value_error(monkeypatch):
    monkeypatch.setattr(main, "filter_companies", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad filter")))

    response = client.get("/filter")

    assert response.status_code == 400
    assert response.json()["detail"] == "bad filter"


def test_company_trademarks_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_company_trademarks",
        lambda inn, limit, offset: {
            "inn": inn,
            "count": 1,
            "total": 1,
            "limit": limit,
            "offset": offset,
            "items": [{"reg_num": "TM-1"}],
        },
    )

    response = client.get("/company/123/trademarks?limit=10&offset=0")

    assert response.status_code == 200
    assert response.json()["inn"] == "123"
    assert response.json()["items"][0]["reg_num"] == "TM-1"


def test_company_mktu_stats_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_company_mktu_stats",
        lambda inn: {
            "inn": inn,
            "unique_classes": [39, 43],
            "classes_count": {"39": 1, "43": 1},
        },
    )

    response = client.get("/company/123/mktu_stats")

    assert response.status_code == 200
    assert response.json()["inn"] == "123"


def test_model_info_success(monkeypatch):
    monkeypatch.setattr(main, "get_model_info", lambda: {"status": "ready"})

    response = client.get("/model_info")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_search_logo_success(monkeypatch, png_bytes):
    monkeypatch.setattr(
        main,
        "find_similar_logos",
        lambda file_bytes, top_k, query_mktu: {
            "results": [
                {"trademark_id": "TM-1", "similarity": 0.95},
                {"trademark_id": "TM-2", "similarity": 0.90},
            ],
            "risk": {"level": "low"},
        },
    )

    response = client.post(
        "/search_logo",
        files={"file": ("logo.png", png_bytes, "image/png")},
        data={"top_k": 5, "query_mktu": "39,43"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "logo.png"
    assert data["content_type"] == "image/png"
    assert data["top_k"] == 5
    assert "results" in data


def test_search_logo_invalid_image(monkeypatch):
    monkeypatch.setattr(
        main,
        "find_similar_logos",
        lambda file_bytes, top_k, query_mktu: (_ for _ in ()).throw(main.InvalidImageError("bad image")),
    )

    response = client.post(
        "/search_logo",
        files={"file": ("bad.txt", b"hello", "text/plain")},
        data={"top_k": 5, "query_mktu": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad image"


def test_search_logo_model_execution_error(monkeypatch, png_bytes):
    monkeypatch.setattr(
        main,
        "find_similar_logos",
        lambda file_bytes, top_k, query_mktu: (_ for _ in ()).throw(main.ModelExecutionError("model failed")),
    )

    response = client.post(
        "/search_logo",
        files={"file": ("logo.png", png_bytes, "image/png")},
        data={"top_k": 5, "query_mktu": ""},
    )

    assert response.status_code == 500
    assert "Ошибка выполнения модели" in response.json()["detail"]


def test_search_logo_model_result_error(monkeypatch, png_bytes):
    monkeypatch.setattr(
        main,
        "find_similar_logos",
        lambda file_bytes, top_k, query_mktu: (_ for _ in ()).throw(main.ModelResultError("bad result")),
    )

    response = client.post(
        "/search_logo",
        files={"file": ("logo.png", png_bytes, "image/png")},
        data={"top_k": 5, "query_mktu": ""},
    )

    assert response.status_code == 500
    assert "Ошибка обработки результатов модели" in response.json()["detail"]


def test_search_logo_unexpected_error(monkeypatch, png_bytes):
    monkeypatch.setattr(
        main,
        "find_similar_logos",
        lambda file_bytes, top_k, query_mktu: (_ for _ in ()).throw(Exception("unexpected")),
    )

    response = client.post(
        "/search_logo",
        files={"file": ("logo.png", png_bytes, "image/png")},
        data={"top_k": 5, "query_mktu": ""},
    )

    assert response.status_code == 500
    assert "Неожиданная ошибка при поиске похожих логотипов" in response.json()["detail"]