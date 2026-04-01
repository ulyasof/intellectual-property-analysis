import numpy as np
import pandas as pd
import pytest

import data_loader


def test_parse_classes_comma_string():
    assert data_loader._parse_classes("39,43") == [39, 43]


def test_parse_classes_string_with_spaces():
    assert data_loader._parse_classes("10, 27, 32") == [10, 27, 32]


def test_parse_classes_number():
    assert data_loader._parse_classes(44) == [44]


def test_parse_classes_nan():
    assert data_loader._parse_classes(np.nan) == []


def test_parse_classes_invalid_string():
    assert data_loader._parse_classes("abc, 39, x, 12") == [39, 12]


def test_normalize_value_nan():
    assert data_loader._normalize_value(np.nan) is None


def test_normalize_value_numpy_scalar():
    value = np.int64(7)
    result = data_loader._normalize_value(value)
    assert result == 7
    assert isinstance(result, int)


def test_normalize_record():
    record = {
        "a": np.nan,
        "b": np.int64(5),
        "c": "text",
    }
    result = data_loader._normalize_record(record)
    assert result == {"a": None, "b": 5, "c": "text"}


def test_get_companies_default(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_companies()

    assert result["count"] == 3
    assert result["total"] == 3
    assert result["limit"] == 100
    assert result["offset"] == 0
    assert len(result["items"]) == 3


def test_get_companies_with_limit_offset(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_companies(limit=1, offset=1)

    assert result["count"] == 1
    assert result["total"] == 3
    assert result["items"][0]["inn"] == "456"


def test_get_companies_with_columns(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_companies(columns=["inn", "company_name"])

    assert result["count"] == 3
    assert result["columns"] == ["inn", "company_name"]
    assert set(result["items"][0].keys()) == {"inn", "company_name"}


def test_get_companies_invalid_columns(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    with pytest.raises(ValueError, match="В датасете отсутствуют колонки"):
        data_loader.get_companies(columns=["bad_column"])


def test_get_company_by_inn_found(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_company_by_inn("123")

    assert result is not None
    assert result["inn"] == "123"
    assert result["company_name"] == "Alpha"


def test_get_company_by_inn_not_found(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_company_by_inn("000")
    assert result is None


def test_get_company_by_inn_without_inn_column(monkeypatch):
    df = pd.DataFrame([{"company_name": "OnlyName"}])
    monkeypatch.setattr(data_loader, "load_dataset", lambda: df)

    with pytest.raises(ValueError, match="колонка 'inn'"):
        data_loader.get_company_by_inn("123")


def test_get_companies_short(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_companies_short(limit=2, offset=0)

    assert result["count"] == 2
    assert "inn" in result["columns"]
    assert "company_name" in result["columns"]
    assert "brand_score" in result["columns"]


def test_get_cluster_stats(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_cluster_stats()

    assert result == [
        {"cluster": 0, "count": 1},
        {"cluster": 1, "count": 2},
    ]


def test_get_cluster_stats_without_cluster(monkeypatch):
    df = pd.DataFrame([
        {"inn": "123", "company_name": "Alpha"},
    ])
    monkeypatch.setattr(data_loader, "load_dataset", lambda: df)

    with pytest.raises(ValueError, match="колонка 'cluster'"):
        data_loader.get_cluster_stats()


def test_get_brand_score_distribution(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_brand_score_distribution()

    assert len(result) == 3
    assert result[0]["brand_score"] >= result[1]["brand_score"] >= result[2]["brand_score"]
    assert "inn" in result[0]
    assert "company_name" in result[0]


def test_get_top_companies_by_brand_score(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_top_companies_by_brand_score(limit=2)

    assert len(result) == 2
    assert result[0]["company_name"] == "Alpha"
    assert result[1]["company_name"] == "Gamma"


def test_get_pca_data(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_pca_data()

    assert len(result) == 3
    assert "pca_x" in result[0]
    assert "pca_y" in result[0]


def test_get_pca_data_missing_required_column(monkeypatch):
    df = pd.DataFrame([
        {"pca_x": 1.0, "inn": "123"},
    ])
    monkeypatch.setattr(data_loader, "load_dataset", lambda: df)

    with pytest.raises(ValueError, match="pca_y"):
        data_loader.get_pca_data()


def test_get_numeric_summary(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_numeric_summary()

    assert result["rows_count"] == 3
    assert "brand_score" in result["numeric_columns"]
    assert "stats" in result
    assert "brand_score" in result["stats"]
    assert "min" in result["stats"]["brand_score"]
    assert "max" in result["stats"]["brand_score"]
    assert "mean" in result["stats"]["brand_score"]


def test_get_available_columns(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.get_available_columns()

    assert "inn" in result
    assert "company_name" in result


def test_filter_companies_by_cluster(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.filter_companies(cluster=1)

    assert result["total"] == 2
    assert result["count"] == 2
    assert all(item["cluster"] == 1 for item in result["items"])


def test_filter_companies_by_min_brand_score(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.filter_companies(min_brand_score=0.7)

    assert result["total"] == 2
    assert all(item["brand_score"] >= 0.7 for item in result["items"])


def test_filter_companies_by_max_brand_score(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.filter_companies(max_brand_score=0.6)

    assert result["total"] == 1
    assert all(item["brand_score"] <= 0.6 for item in result["items"])


def test_filter_companies_combined(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.filter_companies(cluster=1, min_brand_score=0.8, max_brand_score=1.0)

    assert result["total"] == 1
    assert result["items"][0]["inn"] == "123"


def test_filter_companies_with_columns(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    result = data_loader.filter_companies(columns=["inn", "brand_score"])

    assert result["count"] == 3
    assert set(result["items"][0].keys()) == {"inn", "brand_score"}


def test_filter_companies_invalid_columns(monkeypatch, sample_companies_df):
    monkeypatch.setattr(data_loader, "load_dataset", lambda: sample_companies_df)

    with pytest.raises(ValueError, match="В датасете отсутствуют колонки"):
        data_loader.filter_companies(columns=["bad_column"])


def test_filter_companies_missing_cluster_column(monkeypatch):
    df = pd.DataFrame([
        {"inn": "123", "brand_score": 0.9},
    ])
    monkeypatch.setattr(data_loader, "load_dataset", lambda: df)

    with pytest.raises(ValueError, match="колонка 'cluster'"):
        data_loader.filter_companies(cluster=1)


def test_filter_companies_missing_brand_score_column(monkeypatch):
    df = pd.DataFrame([
        {"inn": "123", "cluster": 1},
    ])
    monkeypatch.setattr(data_loader, "load_dataset", lambda: df)

    with pytest.raises(ValueError, match="колонка 'brand_score'"):
        data_loader.filter_companies(min_brand_score=0.5)


def test_get_company_trademarks(monkeypatch, sample_trademarks_df):
    monkeypatch.setattr(data_loader, "load_trademarks_dataset", lambda: sample_trademarks_df)

    result = data_loader.get_company_trademarks("123")

    assert result["inn"] == "123"
    assert result["total"] == 3
    assert result["count"] == 3
    assert result["items"][0]["classes_list"] == [39, 43]
    assert result["items"][0]["classes_count"] == 2
    assert result["items"][2]["classes_list"] == []
    assert result["items"][2]["classes_count"] == 0


def test_get_company_trademarks_with_pagination(monkeypatch, sample_trademarks_df):
    monkeypatch.setattr(data_loader, "load_trademarks_dataset", lambda: sample_trademarks_df)

    result = data_loader.get_company_trademarks("123", limit=1, offset=1)

    assert result["total"] == 3
    assert result["count"] == 1
    assert result["items"][0]["reg_num"] == "TM-2"


def test_get_company_trademarks_missing_inn_column(monkeypatch):
    df = pd.DataFrame([
        {"reg_num": "TM-1", "classes": "39,43"},
    ])
    monkeypatch.setattr(data_loader, "load_trademarks_dataset", lambda: df)

    with pytest.raises(ValueError, match="колонка 'inn'"):
        data_loader.get_company_trademarks("123")


def test_get_company_mktu_stats(monkeypatch, sample_trademarks_df):
    monkeypatch.setattr(data_loader, "load_trademarks_dataset", lambda: sample_trademarks_df)

    result = data_loader.get_company_mktu_stats("123")

    assert result["inn"] == "123"
    assert result["unique_classes"] == [10, 27, 32, 39, 43]
    assert result["classes_count"] == {
        39: 1,
        43: 1,
        10: 1,
        27: 1,
        32: 1,
    }


def test_get_company_mktu_stats_no_classes(monkeypatch):
    df = pd.DataFrame([
        {"inn": "123", "reg_num": "TM-1", "classes": None},
    ])
    monkeypatch.setattr(data_loader, "load_trademarks_dataset", lambda: df)

    result = data_loader.get_company_mktu_stats("123")

    assert result == {
        "inn": "123",
        "unique_classes": [],
        "classes_count": {},
    }