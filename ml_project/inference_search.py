from pathlib import Path
import argparse
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import models

from retrieval_dataset import (
    build_eval_transform,
    load_image_rgb_on_white,
)
from risk_utils import (
    estimate_trademark_risk,
    format_mktu_classes,
)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EmbeddingModel(nn.Module):
    def __init__(self, model_name: str = "resnet50", emb_dim: int = 256):
        super().__init__()
        model_name = model_name.lower()

        if model_name == "resnet18":
            base = models.resnet18(weights=None)
            feat_dim = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        elif model_name == "resnet50":
            base = models.resnet50(weights=None)
            feat_dim = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        elif model_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=None)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base

        else:
            raise ValueError(
                f"Unsupported model_name='{model_name}'. "
                f"resnet18, resnet50, efficientnet_b0"
            )

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = feats.flatten(1)
        emb = self.head(feats)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


@torch.no_grad()
def embed_one_image(model, image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    transform = build_eval_transform(image_size=image_size)
    image = load_image_rgb_on_white(image_path)
    x = transform(image).unsqueeze(0).to(device)
    emb = model(x)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu()


def enrich_gallery_metadata(gallery_meta: pd.DataFrame, fallback_csv: Path) -> pd.DataFrame:
    needed_cols = ["brand_name", "mktu_classes", "status"]

    if all(col in gallery_meta.columns for col in needed_cols):
        return gallery_meta.fillna("")

    if not fallback_csv.exists():
        for col in needed_cols:
            if col not in gallery_meta.columns:
                gallery_meta[col] = ""
        return gallery_meta.fillna("")

    fallback_df = pd.read_csv(fallback_csv, dtype=str).fillna("")
    merge_cols = [col for col in ["image_id", "tm_id", "path"] if col in gallery_meta.columns and col in fallback_df.columns]
    extra_cols = [col for col in needed_cols if col in fallback_df.columns]

    if not merge_cols or not extra_cols:
        for col in needed_cols:
            if col not in gallery_meta.columns:
                gallery_meta[col] = ""
        return gallery_meta.fillna("")

    merged = gallery_meta.merge(
        fallback_df[merge_cols + extra_cols].drop_duplicates(subset=merge_cols),
        on=merge_cols,
        how="left",
        suffixes=("", "_fallback"),
    )

    for col in needed_cols:
        fallback_col = f"{col}_fallback"
        if col not in merged.columns:
            merged[col] = ""
        if fallback_col in merged.columns:
            merged[col] = merged[col].astype(str).replace("nan", "")
            merged[fallback_col] = merged[fallback_col].astype(str).replace("nan", "")
            merged[col] = merged[col].where(merged[col].astype(str).str.strip() != "", merged[fallback_col])
            merged = merged.drop(columns=[fallback_col])

    return merged.fillna("")


def make_result_visualization(
    query_path: Path,
    results_df: pd.DataFrame,
    project_root: Path,
    risk_result: dict,
    query_mktu: str,
    save_path: Path,
):
    cell_w = 290
    cell_h = 280
    text_h = 120
    header_h = 110
    cols = 3
    bg = "white"
    font_fill = "black"

    items = [
        ("QUERY", str(query_path), f"Входное изображение\nМКТУ: {format_mktu_classes(query_mktu)}")
    ]

    for rank, row in results_df.iterrows():
        title = f"TOP{rank + 1}"
        subtitle = (
            f"tm_id={row['tm_id']} | score={row['score']:.4f}\n"
            f"МКТУ: {format_mktu_classes(row.get('mktu_classes', ''))}"
        )
        items.append((title, row["path"], subtitle))

    rows = (len(items) + cols - 1) // cols
    canvas_h = header_h + rows * (cell_h + text_h)
    canvas = Image.new("RGB", (cols * cell_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    header_text = (
        f"Оценка риска: {risk_result['risk_score']}/100 | уровень: {risk_result['risk_level']}\n"
        f"Наиболее конфликтный знак: {risk_result.get('top_conflict_tm_id', '')}\n"
        f"Пересечение МКТУ: {', '.join(risk_result.get('mktu_overlap', [])) if risk_result.get('mktu_overlap') else 'нет'}"
    )
    draw.rectangle([0, 0, cols * cell_w, header_h], fill="white")
    draw.multiline_text((15, 12), header_text, fill=font_fill, spacing=6)

    for i, (title, img_path_str, subtitle) in enumerate(items):
        c = i % cols
        r = i // cols
        x = c * cell_w
        y = header_h + r * (cell_h + text_h)

        img_path = Path(img_path_str)
        if not img_path.is_absolute():
            img_path = project_root / img_path

        img = load_image_rgb_on_white(img_path)
        img.thumbnail((cell_w - 20, cell_h - 20))

        tile = Image.new("RGB", (cell_w, cell_h), bg)
        px = (cell_w - img.width) // 2
        py = (cell_h - img.height) // 2
        tile.paste(img, (px, py))
        canvas.paste(tile, (x, y))

        draw.rectangle([x, y + cell_h, x + cell_w, y + cell_h + text_h], fill="white")
        draw.text((x + 10, y + cell_h + 8), title, fill=font_fill)
        draw.multiline_text((x + 10, y + cell_h + 32), subtitle, fill=font_fill, spacing=5)

    canvas.save(save_path, quality=95)


def build_unique_topk_results(
    sims: torch.Tensor,
    gallery_meta: pd.DataFrame,
    top_k: int,
) -> list[dict]:
    sorted_indices = torch.argsort(sims, descending=True).tolist()

    seen_tm_ids = set()
    results = []

    for idx in sorted_indices:
        row = gallery_meta.iloc[idx]
        tm_id = str(row.get("tm_id", "")).strip()

        if tm_id in seen_tm_ids:
            continue

        seen_tm_ids.add(tm_id)

        results.append({
            "rank": len(results) + 1,
            "tm_id": tm_id,
            "image_id": row.get("image_id", ""),
            "path": row.get("path", ""),
            "source_type": row.get("source_type", ""),
            "brand_name": row.get("brand_name", ""),
            "mktu_classes": row.get("mktu_classes", ""),
            "status": row.get("status", ""),
            "score": float(sims[idx].item()),
        })

        if len(results) >= top_k:
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Search top-k similar trademarks with risk estimation.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Путь к входной картинке"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="final_model",
        help="Папка с best_model.pt, gallery_embeddings.pt и gallery_metadata.csv"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Корень проекта"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Сколько ближайших разных tm_id вернуть"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Куда сохранить csv/json и визуализацию"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto / cpu / cuda / mps"
    )
    parser.add_argument(
        "--query_mktu_classes",
        type=str,
        default="",
        help="Классы МКТУ для нового знака, например: '25 35'"
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image_path)
    if not image_path.is_absolute():
        image_path = project_root / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    checkpoint_path = model_dir / "best_model.pt"
    gallery_embeddings_path = model_dir / "gallery_embeddings.pt"
    gallery_metadata_path = model_dir / "gallery_metadata.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not gallery_embeddings_path.exists():
        raise FileNotFoundError(f"Missing gallery embeddings: {gallery_embeddings_path}")
    if not gallery_metadata_path.exists():
        raise FileNotFoundError(f"Missing gallery metadata: {gallery_metadata_path}")

    device = choose_device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint["model_name"]
    emb_dim = checkpoint["emb_dim"]
    image_size = checkpoint["image_size"]

    model = EmbeddingModel(model_name=model_name, emb_dim=emb_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    gallery_embeddings = torch.load(gallery_embeddings_path, map_location="cpu")
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    gallery_meta = pd.read_csv(gallery_metadata_path, dtype=str).fillna("")
    gallery_meta = enrich_gallery_metadata(
        gallery_meta=gallery_meta,
        fallback_csv=project_root / "prepared" / "final_retrieval" / "gallery_registry.csv",
    )

    query_emb = embed_one_image(model, image_path, image_size=image_size, device=device)
    sims = (query_emb @ gallery_embeddings.T).squeeze(0)

    top_k = min(args.top_k, len(gallery_meta))
    results = build_unique_topk_results(
        sims=sims,
        gallery_meta=gallery_meta,
        top_k=top_k,
    )

    results_df = pd.DataFrame(results)
    risk_result = estimate_trademark_risk(
        results_df=results_df,
        query_mktu=args.query_mktu_classes,
    )

    stem = image_path.stem
    csv_path = output_dir / f"{stem}_top{top_k}.csv"
    json_path = output_dir / f"{stem}_top{top_k}.json"
    risk_path = output_dir / f"{stem}_risk.json"
    vis_path = output_dir / f"{stem}_top{top_k}.jpg"

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(risk_path, "w", encoding="utf-8") as f:
        json.dump(risk_result, f, ensure_ascii=False, indent=2)

    make_result_visualization(
        query_path=image_path,
        results_df=results_df,
        project_root=project_root,
        risk_result=risk_result,
        query_mktu=args.query_mktu_classes,
        save_path=vis_path,
    )


    print("Inference finished")
    print(f"Input image: {image_path}")
    print(f"Model dir:   {model_dir}")
    print(f"Device:      {device}")
    print(f"Risk score:  {risk_result['risk_score']}/100")
    print(f"Risk level:  {risk_result['risk_level']}")
    print()
    print("Top results:")
    for row in results:
        print(
            f"TOP{row['rank']}: "
            f"tm_id={row['tm_id']} | "
            f"score={row['score']:.4f} | "
            f"МКТУ={format_mktu_classes(row.get('mktu_classes', ''))} | "
            f"path={row['path']}"
        )
    print()
    print("Saved files:")
    print(f" - {csv_path}")
    print(f" - {json_path}")
    print(f" - {risk_path}")
    print(f" - {vis_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()