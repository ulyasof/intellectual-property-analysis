import io
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def sample_companies_df():
    return pd.DataFrame([
        {
            "inn": "123",
            "company_name": "Alpha",
            "brand_score": 0.91,
            "cluster": 1,
            "revenue": 1000,
            "trademarks_count": 3,
            "pca_x": 1.5,
            "pca_y": 2.5,
        },
        {
            "inn": "456",
            "company_name": "Beta",
            "brand_score": 0.55,
            "cluster": 0,
            "revenue": 2000,
            "trademarks_count": 1,
            "pca_x": -0.5,
            "pca_y": 0.3,
        },
        {
            "inn": "789",
            "company_name": "Gamma",
            "brand_score": 0.72,
            "cluster": 1,
            "revenue": None,
            "trademarks_count": 2,
            "pca_x": 3.1,
            "pca_y": -1.2,
        },
    ])


@pytest.fixture
def sample_trademarks_df():
    return pd.DataFrame([
        {"inn": "123", "reg_num": "TM-1", "classes": "39,43"},
        {"inn": "123", "reg_num": "TM-2", "classes": "10, 27, 32"},
        {"inn": "123", "reg_num": "TM-3", "classes": None},
        {"inn": "456", "reg_num": "TM-4", "classes": 44},
    ])


@pytest.fixture
def png_bytes():
    image = Image.new("RGB", (10, 10), color="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes():
    image = Image.new("RGB", (10, 10), color="white")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()