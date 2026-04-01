import io
import json
from pathlib import Path

import pytest
from PIL import Image

import model_adapter


def make_webp_bytes():
    image = Image.new("RGB", (10, 10), color="white")
    buf = io.BytesIO()
    image.save(buf, format="WEBP")
    return buf.getvalue()


def test_validate_image_bytes_png(png_bytes):
    suffix = model_adapter.validate_image_bytes(png_bytes)
    assert suffix == ".png"


def test_validate_image_bytes_jpeg(jpeg_bytes):
    suffix = model_adapter.validate_image_bytes(jpeg_bytes)
    assert suffix == ".jpg"


def test_validate_image_bytes_webp():
    file_bytes = make_webp_bytes()
    suffix = model_adapter.validate_image_bytes(file_bytes)
    assert suffix == ".webp"


def test_validate_image_bytes_empty():
    with pytest.raises(model_adapter.InvalidImageError, match="Файл пустой"):
        model_adapter.validate_image_bytes(b"")


def test_validate_image_bytes_too_large(monkeypatch):
    fake_bytes = b"a" * (model_adapter.MAX_IMAGE_SIZE + 1)

    with pytest.raises(model_adapter.InvalidImageError, match="Файл слишком большой"):
        model_adapter.validate_image_bytes(fake_bytes)


def test_validate_image_bytes_not_image():
    with pytest.raises(model_adapter.InvalidImageError, match="не является корректным изображением"):
        model_adapter.validate_image_bytes(b"not an image")


def test_validate_image_bytes_unsupported_format():
    image = Image.new("RGB", (10, 10), color="white")
    buf = io.BytesIO()
    image.save(buf, format="BMP")
    bmp_bytes = buf.getvalue()

    with pytest.raises(model_adapter.InvalidImageError, match="Неподдерживаемый формат изображения"):
        model_adapter.validate_image_bytes(bmp_bytes)


def test_build_image_url_ok():
    path = "/root/project/model_handoff/dataset/class_a/image1.png"
    base_url = "http://localhost:8000"

    result = model_adapter.build_image_url(path, base_url)

    assert result == "http://localhost:8000/dataset/class_a/image1.png"


def test_build_image_url_empty():
    assert model_adapter.build_image_url("", "http://localhost:8000") is None


def test_build_image_url_without_dataset():
    path = "/root/project/images/image1.png"
    result = model_adapter.build_image_url(path, "http://localhost:8000")
    assert result is None


def test_get_model_info_ready(monkeypatch):
    class FakePath:
        def exists(self):
            return True

        def __str__(self):
            return "model_handoff/inference_search.py"

    monkeypatch.setattr(model_adapter, "Path", lambda *args, **kwargs: FakePath())

    result = model_adapter.get_model_info()

    assert result["model_type"] == "external_inference"
    assert result["status"] == "ready"


def test_get_model_info_missing(monkeypatch):
    class FakePath:
        def exists(self):
            return False

        def __str__(self):
            return "model_handoff/inference_search.py"

    monkeypatch.setattr(model_adapter, "Path", lambda *args, **kwargs: FakePath())

    result = model_adapter.get_model_info()

    assert result["status"] == "missing"


def test_find_similar_logos_success(monkeypatch, tmp_path, png_bytes):
    project_root = tmp_path
    output_dir = project_root / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path_holder = {}
    risk_path_holder = {}

    real_path_class = Path

    class FakeCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = "ok"
            self.stderr = ""

    def fake_run(command, capture_output, text, check):
        top_k = command[command.index("--top_k") + 1]
        out_dir = Path(command[command.index("--output_dir") + 1])
        image_path = Path(command[command.index("--image_path") + 1])
        unique_name = image_path.stem

        json_path = out_dir / f"{unique_name}_top{top_k}.json"
        risk_path = out_dir / f"{unique_name}_risk.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([
                {
                    "tm_id": "TM-1",
                    "score": 0.98765,
                    "image_id": "IMG-1",
                    "path": str(project_root / "model_handoff" / "dataset" / "logos" / "img1.png"),
                    "source_type": "dataset",
                    "rank": 1,
                    "mktu_classes": [39, 43],
                }
            ], f, ensure_ascii=False)

        with open(risk_path, "w", encoding="utf-8") as f:
            json.dump({"level": "low"}, f, ensure_ascii=False)

        json_path_holder["path"] = json_path
        risk_path_holder["path"] = risk_path

        return FakeCompletedProcess()

    monkeypatch.setattr(model_adapter, "subprocess", type("FakeSubprocess", (), {"run": staticmethod(fake_run)}))

    def fake_resolve(self):
        return self

    monkeypatch.setattr(model_adapter.Path, "resolve", fake_resolve, raising=False)

    original_file = model_adapter.__file__
    monkeypatch.setattr(model_adapter, "__file__", str(project_root / "model_adapter.py"))

    result = model_adapter.find_similar_logos(file_bytes=png_bytes, top_k=5, query_mktu="39,43")

    assert "results" in result
    assert "risk" in result
    assert result["risk"] == {"level": "low"}
    assert len(result["results"]) == 1
    assert result["results"][0]["trademark_id"] == "TM-1"
    assert result["results"][0]["similarity"] == 0.9877
    assert result["results"][0]["image_url"] == "http://194.67.102.116:8000/dataset/logos/img1.png"

    monkeypatch.setattr(model_adapter, "__file__", original_file)


def test_find_similar_logos_model_execution_error(monkeypatch, tmp_path, png_bytes):
    class FakeCompletedProcess:
        returncode = 1
        stdout = "some stdout"
        stderr = "some stderr"

    def fake_run(command, capture_output, text, check):
        return FakeCompletedProcess()

    monkeypatch.setattr(model_adapter, "subprocess", type("FakeSubprocess", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(model_adapter, "__file__", str(tmp_path / "model_adapter.py"))

    with pytest.raises(model_adapter.ModelExecutionError, match="Ошибка запуска модели"):
        model_adapter.find_similar_logos(file_bytes=png_bytes, top_k=5, query_mktu="")


def test_find_similar_logos_missing_json(monkeypatch, tmp_path, png_bytes):
    class FakeCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, capture_output, text, check):
        return FakeCompletedProcess()

    monkeypatch.setattr(model_adapter, "subprocess", type("FakeSubprocess", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(model_adapter, "__file__", str(tmp_path / "model_adapter.py"))

    with pytest.raises(model_adapter.ModelResultError, match="JSON-результат не найден"):
        model_adapter.find_similar_logos(file_bytes=png_bytes, top_k=5, query_mktu="")


def test_find_similar_logos_missing_risk_json(monkeypatch, tmp_path, png_bytes):
    project_root = tmp_path
    output_dir = project_root / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    class FakeCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, capture_output, text, check):
        out_dir = Path(command[command.index("--output_dir") + 1])
        image_path = Path(command[command.index("--image_path") + 1])
        top_k = command[command.index("--top_k") + 1]
        unique_name = image_path.stem

        json_path = out_dir / f"{unique_name}_top{top_k}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        return FakeCompletedProcess()

    monkeypatch.setattr(model_adapter, "subprocess", type("FakeSubprocess", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(model_adapter, "__file__", str(project_root / "model_adapter.py"))

    with pytest.raises(model_adapter.ModelResultError, match="JSON-результат риска не найден"):
        model_adapter.find_similar_logos(file_bytes=png_bytes, top_k=5, query_mktu="")


def test_find_similar_logos_invalid_main_json(monkeypatch, tmp_path, png_bytes):
    project_root = tmp_path
    output_dir = project_root / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    class FakeCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, capture_output, text, check):
        out_dir = Path(command[command.index("--output_dir") + 1])
        image_path = Path(command[command.index("--image_path") + 1])
        top_k = command[command.index("--top_k") + 1]
        unique_name = image_path.stem

        json_path = out_dir / f"{unique_name}_top{top_k}.json"
        risk_path = out_dir / f"{unique_name}_risk.json"

        with open(json_path, "w", encoding="utf-8") as f:
            f.write("{bad json")

        with open(risk_path, "w", encoding="utf-8") as f:
            json.dump({"level": "low"}, f)

        return FakeCompletedProcess()

    monkeypatch.setattr(model_adapter, "subprocess", type("FakeSubprocess", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(model_adapter, "__file__", str(project_root / "model_adapter.py"))

    with pytest.raises(model_adapter.ModelResultError, match="Не удалось прочитать JSON-результат модели"):
        model_adapter.find_similar_logos(file_bytes=png_bytes, top_k=5, query_mktu="")


def test_find_similar_logos_invalid_risk_json(monkeypatch, tmp_path, png_bytes):
    project_root = tmp_path
    output_dir = project_root / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    class FakeCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, capture_output, text, check):
        out_dir = Path(command[command.index("--output_dir") + 1])
        image_path = Path(command[command.index("--image_path") + 1])
        top_k = command[command.index("--top_k") + 1]
        unique_name = image_path.stem

        json_path = out_dir / f"{unique_name}_top{top_k}.json"
        risk_path = out_dir / f"{unique_name}_risk.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        with open(risk_path, "w", encoding="utf-8") as f:
            f.write("{bad json")

        return FakeCompletedProcess()

    monkeypatch.setattr(model_adapter, "subprocess", type("FakeSubprocess", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(model_adapter, "__file__", str(project_root / "model_adapter.py"))

    with pytest.raises(model_adapter.ModelResultError, match="Не удалось прочитать JSON-результат риска"):
        model_adapter.find_similar_logos(file_bytes=png_bytes, top_k=5, query_mktu="")