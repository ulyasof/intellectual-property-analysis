import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# подставляет белый фон
def load_image_rgb_on_white(image_path: Path) -> Image.Image:
    img = Image.open(image_path)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img)
        img = img.convert("RGB")
    else:
        img = img.convert("RGB")

    return img


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# делает квадрат 
class SquarePad:
    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img

        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_top = (max_side - h) // 2
        pad_right = max_side - w - pad_left
        pad_bottom = max_side - h - pad_top

        return TF.pad(
            img,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill
        )
    

def build_transform(image_size: int = 224):
    return transforms.Compose([
        SquarePad(fill=(255, 255, 255)),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_encoder(model_name: str = "resnet50", weights_path: str | None = None) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        encoder = nn.Sequential(*list(model.children())[:-1])

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        encoder = nn.Sequential(*list(model.children())[:-1])

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        model.classifier = nn.Identity()
        encoder = model

    else:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            f"Use one of: resnet18, resnet50, efficientnet_b0"
        )

    encoder.eval()
    return encoder


def resolve_image_path(project_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return project_root / p


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: Path, project_root: Path, transform):
        self.df = pd.read_csv(csv_path, dtype=str).fillna("")
        self.project_root = project_root
        self.transform = transform

        required_cols = {"image_id", "tm_id", "path", "source_type"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve_image_path(self.project_root, row["path"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = load_image_rgb_on_white(img_path)
        image = self.transform(image)

        meta = {
            "image_id": row["image_id"],
            "tm_id": row["tm_id"],
            "path": row["path"],
            "source_type": row["source_type"],
        }
        return image, meta


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    metas = [item[1] for item in batch]
    return images, metas


@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    all_embeddings = []
    all_meta = []

    for images, metas in dataloader:
        images = images.to(device)
        feats = encoder(images)

        if feats.ndim == 4:
            feats = feats.flatten(1)

        feats = F.normalize(feats, p=2, dim=1)

        all_embeddings.append(feats.cpu())
        all_meta.extend(metas)

    embeddings = torch.cat(all_embeddings, dim=0)
    meta_df = pd.DataFrame(all_meta)
    return embeddings, meta_df


def evaluate_fold(
    fold_dir: Path,
    project_root: Path,
    encoder: nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    gallery_csv = fold_dir / "gallery_registry.csv"
    val_queries_csv = fold_dir / "val_queries_real.csv"

    if not gallery_csv.exists():
        raise FileNotFoundError(f"Missing file: {gallery_csv}")
    if not val_queries_csv.exists():
        raise FileNotFoundError(f"Missing file: {val_queries_csv}")

    gallery_ds = CsvImageDataset(gallery_csv, project_root, transform)
    query_ds = CsvImageDataset(val_queries_csv, project_root, transform)

    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    query_loader = DataLoader(
        query_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    gallery_embs, gallery_meta = extract_embeddings(encoder, gallery_loader, device)
    query_embs, query_meta = extract_embeddings(encoder, query_loader, device)

    sims = query_embs @ gallery_embs.T  # cosine, because embeddings are normalized

    recall_at_1_hits = 0
    recall_at_5_hits = 0

    top5_rows = []

    gallery_tm_ids = gallery_meta["tm_id"].tolist()
    gallery_image_ids = gallery_meta["image_id"].tolist()
    gallery_paths = gallery_meta["path"].tolist()

    k_max = min(5, sims.shape[1])

    for q_idx in range(sims.shape[0]):
        q_tm_id = query_meta.iloc[q_idx]["tm_id"]
        q_image_id = query_meta.iloc[q_idx]["image_id"]
        q_path = query_meta.iloc[q_idx]["path"]

        scores = sims[q_idx]
        top_scores, top_indices = torch.topk(scores, k=k_max, largest=True)

        top_indices = top_indices.tolist()
        top_scores = top_scores.tolist()

        top_tm_ids = [gallery_tm_ids[i] for i in top_indices]

        hit_at_1 = int(any(tm_id == q_tm_id for tm_id in top_tm_ids[:1]))
        hit_at_5 = int(any(tm_id == q_tm_id for tm_id in top_tm_ids[:5]))

        recall_at_1_hits += hit_at_1
        recall_at_5_hits += hit_at_5

        row = {
            "query_image_id": q_image_id,
            "query_tm_id": q_tm_id,
            "query_path": q_path,
            "hit_at_1": hit_at_1,
            "hit_at_5": hit_at_5,
        }

        for rank, (g_idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            row[f"top{rank}_image_id"] = gallery_image_ids[g_idx]
            row[f"top{rank}_tm_id"] = gallery_tm_ids[g_idx]
            row[f"top{rank}_path"] = gallery_paths[g_idx]
            row[f"top{rank}_score"] = float(score)

        top5_rows.append(row)

    num_queries = len(query_meta)
    recall_at_1 = recall_at_1_hits / num_queries if num_queries > 0 else 0.0
    recall_at_5 = recall_at_5_hits / num_queries if num_queries > 0 else 0.0

    metrics = {
        "fold": fold_dir.name,
        "num_queries": num_queries,
        "gallery_size": len(gallery_meta),
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
    }

    top5_df = pd.DataFrame(top5_rows)
    return metrics, top5_df


def main():
    parser = argparse.ArgumentParser(description="Zero-shot retrieval evaluation on CV folds")
    parser.add_argument(
        "--cv_dir",
        type=str,
        default="prepared/retrieval_cv",
        help="Путь к папке с fold_0, fold_1, "
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Корень проекта, относительно которого разрешаются пути к изображениям"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50", "efficientnet_b0"],
        help="Pretrained encoder"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Размер изображения после resize"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size для извлечения эмбеддингов"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers для DataLoader; на Mac обычно лучше 0"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto / cpu / cuda / mps"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="Если указать, будет оценён только один fold, например: fold_0"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Локальный путь к .pth весам pretrained модели"
    )
    args = parser.parse_args()

    cv_dir = Path(args.cv_dir)
    project_root = Path(args.project_root)
    device = choose_device(args.device)

    if not cv_dir.exists():
        raise FileNotFoundError(f"CV directory not found: {cv_dir}")

    encoder = build_encoder(args.model_name, weights_path=args.weights_path).to(device)    
    transform = build_transform(args.image_size)

    if args.fold is not None:
        fold_dirs = [cv_dir / args.fold]
    else:
        fold_dirs = sorted([p for p in cv_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])

    if not fold_dirs:
        raise ValueError(f"No fold directories found in {cv_dir}")


    print("Zero-shot retrieval evaluation")
    print()
    print(f"CV dir:       {cv_dir}")
    print(f"Project root: {project_root}")
    print(f"Model:        {args.model_name}")
    print(f"Device:       {device}")
    print(f"Folds:        {[p.name for p in fold_dirs]}")

    all_metrics = []

    for fold_dir in fold_dirs:
        print(f"\nEvaluating {fold_dir.name} ...")

        metrics, top5_df = evaluate_fold(
            fold_dir=fold_dir,
            project_root=project_root,
            encoder=encoder,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(fold_dir / "zero_shot_metrics.csv", index=False)
        top5_df.to_csv(fold_dir / "zero_shot_top5_predictions.csv", index=False)

        all_metrics.append(metrics)

        print(
            f"{fold_dir.name}: "
            f"num_queries={metrics['num_queries']}, "
            f"gallery={metrics['gallery_size']}, "
            f"Recall@1={metrics['recall_at_1']:.4f}, "
            f"Recall@5={metrics['recall_at_5']:.4f}"
        )

    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(cv_dir / "zero_shot_cv_metrics.csv", index=False)

    summary = {
        "model_name": args.model_name,
        "num_folds": len(all_metrics),
        "mean_recall_at_1": float(all_metrics_df["recall_at_1"].mean()),
        "std_recall_at_1": float(all_metrics_df["recall_at_1"].std(ddof=0)),
        "mean_recall_at_5": float(all_metrics_df["recall_at_5"].mean()),
        "std_recall_at_5": float(all_metrics_df["recall_at_5"].std(ddof=0)),
    }

    with open(cv_dir / "zero_shot_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


    print("Cross-validation summary")
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    main()