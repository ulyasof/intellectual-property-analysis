import os

USE_MOCK = os.getenv("USE_MOCK", "").lower() == "true"

if USE_MOCK:
    from api import mock_api as backend
else:
    from api import real_backend_api as backend


def get_companies(limit=100, offset=0, columns=None):
    return backend.get_companies(limit=limit, offset=offset, columns=columns)


def get_companies_short(limit=100, offset=0):
    return backend.get_companies_short(limit=limit, offset=offset)


def get_company_by_inn(inn):
    return backend.get_company_by_inn(inn)


def get_company_trademark_classes(inn):
    return backend.get_company_trademark_classes(inn)


def get_company_trademark_classes_agg(inn):
    return backend.get_company_trademark_classes_agg(inn)


def get_cluster_stats():
    return backend.get_cluster_stats()


def get_brand_score_distribution():
    return backend.get_brand_score_distribution()


def get_top_companies_by_brand_score(limit=10):
    return backend.get_top_companies_by_brand_score(limit=limit)


def get_pca_data():
    return backend.get_pca_data()


def get_numeric_summary():
    return backend.get_numeric_summary()


def get_available_columns():
    return backend.get_available_columns()


def filter_companies(cluster=None, min_brand_score=None, max_brand_score=None, limit=100, offset=0):
    return backend.filter_companies(
        cluster=cluster,
        min_brand_score=min_brand_score,
        max_brand_score=max_brand_score,
        limit=limit,
        offset=offset,
    )


def search_similar_logos(file_name=None, top_k=5, query_mktu_classes=None):
    return backend.search_similar_logos(
        file_path=file_name,
        top_k=top_k,
        query_mktu_classes=query_mktu_classes,
    )