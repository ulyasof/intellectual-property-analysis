from typing import Any, Dict, List
from PIL import Image
from io import BytesIO
from pathlib import Path
import uuid
import tempfile
import os
import subprocess
import json
import sys

class ModelAdapterError(Exception):
    """Базовая ошибка адаптера модели."""


class InvalidImageError(ModelAdapterError):
    """Некорректное входное изображение."""


class ModelExecutionError(ModelAdapterError):
    """Ошибка запуска inference-скрипта."""


class ModelResultError(ModelAdapterError):
    """Ошибка чтения или обработки результатов модели."""

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

def validate_image_bytes(file_bytes: bytes) -> str:
    """
    Проверяет корректность загруженного изображения.

    Проверки:
    - файл не пустой
    - файл не слишком большой
    - файл можно открыть как изображение
    - формат поддерживается
    """

    if not file_bytes:
        raise InvalidImageError("Файл пустой")

    if len(file_bytes) > MAX_IMAGE_SIZE:
        raise InvalidImageError("Файл слишком большой (максимум 10MB)")

    try:
        image = Image.open(BytesIO(file_bytes))
    except Exception:
        raise InvalidImageError("Файл не является корректным изображением")

    format_to_suffix = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "WEBP": ".webp",
    }

    if image.format not in format_to_suffix:
        raise InvalidImageError(
            f"Неподдерживаемый формат изображения: {image.format}"
        )

    #Проверка, что изображение можно полностью прочитать
    try:
        image.verify()
    except Exception:
        raise InvalidImageError("Изображение повреждено")

    return format_to_suffix[image.format]

def build_image_url(path_value: str, base_url: str) -> str | None:
    if not path_value:
        return None

    path = Path(path_value)
    parts = path.parts

    if "dataset" not in parts:
        return None

    dataset_index = parts.index("dataset")
    relative_parts = parts[dataset_index + 1:]
    relative_path = "/".join(relative_parts)

    return f"{base_url}/dataset/{relative_path}"


def find_similar_logos(file_bytes: bytes, top_k: int = 5, query_mktu: str = "") -> Dict[str, Any]:
    """
    Запускает внешний inference-скрипт для поиска похожих логотипов.

    Пайплайн:
    - валидирует изображение,
    - сохраняет его во временный файл,
    - запускает inference_search.py,
    - читает JSON с результатами,
    - возвращает top-k найденных логотипов.
    """
    file_suffix = validate_image_bytes(file_bytes)

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"query_{uuid.uuid4().hex}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(file_bytes)
        temp_image_path = Path(tmp.name)

    renamed_temp_image_path = temp_image_path.with_name(f"{unique_name}{file_suffix}")
    os.replace(temp_image_path, renamed_temp_image_path)
    temp_image_path = renamed_temp_image_path

    json_result_path = output_dir / f"{unique_name}_top{top_k}.json"
    risk_result_path = output_dir / f"{unique_name}_risk.json"
    csv_result_path = output_dir / f"{unique_name}_top{top_k}.csv"
    vis_result_path = output_dir / f"{unique_name}_top{top_k}.jpg"

    try:
        project_root = Path(__file__).resolve().parent
        model_handoff_dir = project_root / "model_handoff"
        inference_script = model_handoff_dir / "inference_search.py"
        model_dir = model_handoff_dir / "final_model"

        command = [
            sys.executable,
            str(inference_script),
            "--image_path",
            str(temp_image_path.resolve()),
            "--model_dir",
            str(model_dir.resolve()),
            "--project_root",
            str(project_root.resolve()),
            "--top_k",
            str(top_k),
            "--output_dir",
            str(output_dir.resolve()),
            "--query_mktu_classes",
            query_mktu,
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if process.returncode != 0:
            raise ModelExecutionError(
                "Ошибка запуска модели.\n"
                f"STDOUT:\n{process.stdout}\n\n"
                f"STDERR:\n{process.stderr}"
            )

        if not json_result_path.exists():
            raise ModelResultError(
                f"JSON-результат не найден: {json_result_path}"
            )

        if not risk_result_path.exists():
            raise ModelResultError(
                f"JSON-результат риска не найден: {risk_result_path}"
            )

        try:
            with open(json_result_path, "r", encoding="utf-8") as f:
                raw_results = json.load(f)
        except Exception as e:
            raise ModelResultError(f"Не удалось прочитать JSON-результат модели: {e}")

        try:
            with open(risk_result_path, "r", encoding="utf-8") as f:
                risk_result = json.load(f)
        except Exception as e:
            raise ModelResultError(f"Не удалось прочитать JSON-результат риска: {e}")

        results = []
        base_url = "http://194.67.102.116:8000"
        for item in raw_results:
            image_url = build_image_url(item.get("path", ""), base_url)
            results.append(
                {
                    "trademark_id": item.get("tm_id"),
                    "similarity": round(float(item.get("score", 0.0)), 4),
                    "company_name": None,
                    "image_id": item.get("image_id"),
                    "path": item.get("path"),
                    "image_url": image_url,
                    "source_type": item.get("source_type"),
                    "rank": item.get("rank"),
                    "mktu_classes": item.get("mktu_classes")
                }
            )

        return {
            "results": results,
            "risk": risk_result,
        }

    finally:
        if temp_image_path.exists():
            temp_image_path.unlink(missing_ok=True)

        json_result_path.unlink(missing_ok=True)
        risk_result_path.unlink(missing_ok=True)
        csv_result_path.unlink(missing_ok=True)
        vis_result_path.unlink(missing_ok=True)


def get_model_info() -> Dict[str, Any]:
    """
    Возвращает информацию о текущем состоянии модели.

    Это удобно для endpoint'а /model_info,
    чтобы в интерфейсе показать,
    что сейчас подключено: заглушка или реальная модель.
    """
    script_path = Path("model_handoff/inference_search.py")

    return {
        "model_type": "external_inference",
        "model_name": "logo_similarity_search",
        "status": "ready" if script_path.exists() else "missing",
        "inference_script": str(script_path),
        "description": "Поиск похожих логотипов через внешний inference script",
        "top_k_default": 5,
        "top_k_max": 20,
    }