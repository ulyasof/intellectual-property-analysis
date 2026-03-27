import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from retrieval_dataset import (
    build_datasets_for_fold,
    build_eval_transform,
    resolve_image_path,
    load_image_rgb_on_white,
)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# для валидации  отдельно читаем val_queries_real.csv и gallery_registry.csv
class SingleImageCsvDataset(Dataset):
    def __init__(self, csv_path: str | Path, project_root: str | Path = ".", transform=None):
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.df = pd.read_csv(self.csv_path, dtype=str).fillna("")
        self.transform = transform

        required_cols = {"image_id", "tm_id", "path", "source_type"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_path}: missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve_image_path(self.project_root, row["path"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = load_image_rgb_on_white(img_path)
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_id": row["image_id"],
            "tm_id": row["tm_id"],
            "path": row["path"],
            "source_type": row["source_type"],
        }

# превращает изобр в векторы
class EmbeddingModel(nn.Module):
    def __init__(self, model_name: str = "resnet50", emb_dim: int = 256, weights_path: str | None = None):
        super().__init__()
        model_name = model_name.lower()

        if model_name == "resnet18":
            base = models.resnet18(weights=None)
            if weights_path is not None:
                state_dict = torch.load(weights_path, map_location="cpu")
                base.load_state_dict(state_dict)
            feat_dim = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        elif model_name == "resnet50":
            base = models.resnet50(weights=None)
            if weights_path is not None:
                state_dict = torch.load(weights_path, map_location="cpu")
                base.load_state_dict(state_dict)
            feat_dim = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        elif model_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=None)
            if weights_path is not None:
                state_dict = torch.load(weights_path, map_location="cpu")
                base.load_state_dict(state_dict)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base

        else:
            raise ValueError(
                f"Unsupported model_name='{model_name}'. "
                f"Use one of: resnet18, resnet50, efficientnet_b0"
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



class MultiPositiveCrossModalLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb: torch.Tensor, target_emb: torch.Tensor, tm_ids: list[str]) -> torch.Tensor:
        logits = (query_emb @ target_emb.T) / self.temperature

        positive_mask = torch.tensor(
            [[a == b for b in tm_ids] for a in tm_ids],
            device=logits.device,
            dtype=torch.bool,
        )

        very_neg = -1e9

        # query -> target
        pos_logits_q = logits.masked_fill(~positive_mask, very_neg)
        log_pos_q = torch.logsumexp(pos_logits_q, dim=1)
        log_all_q = torch.logsumexp(logits, dim=1)
        loss_q = -(log_pos_q - log_all_q).mean()

        # target -> query
        logits_t = logits.T
        pos_logits_t = logits_t.masked_fill(~positive_mask, very_neg)
        log_pos_t = torch.logsumexp(pos_logits_t, dim=1)
        log_all_t = torch.logsumexp(logits_t, dim=1)
        loss_t = -(log_pos_t - log_all_t).mean()

        return 0.5 * (loss_q + loss_t)


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    all_embs = []
    all_rows = []

    for batch in dataloader:
        images = batch["image"].to(device)
        embs = model(images)
        all_embs.append(embs.cpu())

        batch_size = images.shape[0]
        for i in range(batch_size):
            all_rows.append({
                "image_id": batch["image_id"][i],
                "tm_id": batch["tm_id"][i],
                "path": batch["path"][i],
                "source_type": batch["source_type"][i],
            })

    embs = torch.cat(all_embs, dim=0)
    meta_df = pd.DataFrame(all_rows)
    return embs, meta_df


@torch.no_grad()
def evaluate_retrieval_on_fold(model, fold_dir: Path, project_root: Path, device, batch_size: int, num_workers: int, image_size: int):
    transform = build_eval_transform(image_size=image_size)

    queries_csv = fold_dir / "val_queries_real.csv"
    gallery_csv = fold_dir / "gallery_registry.csv"

    query_ds = SingleImageCsvDataset(queries_csv, project_root=project_root, transform=transform)
    gallery_ds = SingleImageCsvDataset(gallery_csv, project_root=project_root, transform=transform)

    query_loader = DataLoader(
        query_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    query_embs, query_meta = extract_embeddings(model, query_loader, device)
    gallery_embs, gallery_meta = extract_embeddings(model, gallery_loader, device)

    sims = query_embs @ gallery_embs.T

    gallery_tm_ids = gallery_meta["tm_id"].tolist()
    gallery_image_ids = gallery_meta["image_id"].tolist()
    gallery_paths = gallery_meta["path"].tolist()

    recall_at_1_hits = 0
    recall_at_5_hits = 0
    top5_rows = []

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

        hit_at_1 = int(any(tm == q_tm_id for tm in top_tm_ids[:1]))
        hit_at_5 = int(any(tm == q_tm_id for tm in top_tm_ids[:5]))

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
        "num_queries": num_queries,
        "gallery_size": len(gallery_meta),
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
    }

    top5_df = pd.DataFrame(top5_rows)
    return metrics, top5_df


def train_one_fold(
    fold_dir: Path,
    project_root: Path,
    output_root: Path,
    device: torch.device,
    model_name: str,
    weights_path: str | None,
    image_size: int,
    emb_dim: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    temperature: float,
):
    fold_name = fold_dir.name
    fold_out = output_root / fold_name
    fold_out.mkdir(parents=True, exist_ok=True)

    train_ds, _ = build_datasets_for_fold(
        fold_dir=fold_dir,
        project_root=project_root,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = EmbeddingModel(
        model_name=model_name,
        emb_dim=emb_dim,
        weights_path=weights_path,
    ).to(device)

    criterion = MultiPositiveCrossModalLoss(temperature=temperature)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.head.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    history = []
    best_score = -1.0
    best_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            query_img = batch["query_image"].to(device)
            target_img = batch["target_image"].to(device)
            tm_ids = list(batch["tm_id"])

            query_emb = model(query_img)
            target_emb = model(target_img)

            loss = criterion(query_emb, target_emb, tm_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        train_loss = epoch_loss / max(1, num_batches)

        val_metrics, top5_df = evaluate_retrieval_on_fold(
            model=model,
            fold_dir=fold_dir,
            project_root=project_root,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_recall_at_1": val_metrics["recall_at_1"],
            "val_recall_at_5": val_metrics["recall_at_5"],
            "num_queries": val_metrics["num_queries"],
            "gallery_size": val_metrics["gallery_size"],
        }
        history.append(row)

        current_score = val_metrics["recall_at_5"]

        print(
            f"[{fold_name}] epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"R@1={val_metrics['recall_at_1']:.4f} | "
            f"R@5={val_metrics['recall_at_5']:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_metrics = {
                "fold": fold_name,
                "best_epoch": epoch,
                "best_val_recall_at_1": val_metrics["recall_at_1"],
                "best_val_recall_at_5": val_metrics["recall_at_5"],
                "num_queries": val_metrics["num_queries"],
                "gallery_size": val_metrics["gallery_size"],
            }

            checkpoint = {
                "epoch": epoch,
                "model_name": model_name,
                "emb_dim": emb_dim,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_recall_at_5": best_score,
                "image_size": image_size,
            }
            torch.save(checkpoint, fold_out / "best_model.pt")
            top5_df.to_csv(fold_out / "best_val_top5_predictions.csv", index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(fold_out / "history.csv", index=False)

    with open(fold_out / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train retrieval model with cross-validation folds.")
    parser.add_argument("--cv_dir", type=str, default="prepared/retrieval_cv")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="checkpoints_metric")
    parser.add_argument("--fold", type=str, default=None, help="Например: fold_0. Если не указан, обучатся все fold.")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet18", "resnet50", "efficientnet_b0"])
    parser.add_argument("--weights_path", type=str, default=None, help="Локальный путь к pretrained .pth")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_backbone", type=float, default=3e-6)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / mps")
    args = parser.parse_args()

    cv_dir = Path(args.cv_dir)
    project_root = Path(args.project_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)

    if args.fold is not None:
        fold_dirs = [cv_dir / args.fold]
    else:
        fold_dirs = sorted([p for p in cv_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])

    if not fold_dirs:
        raise ValueError(f"No fold directories found in {cv_dir}")


    print("Train retrieval model")
    print(f"CV dir:       {cv_dir}")
    print(f"Project root: {project_root}")
    print(f"Output dir:   {output_root}")
    print(f"Model:        {args.model_name}")
    print(f"Device:       {device}")
    print(f"Folds:        {[p.name for p in fold_dirs]}")


    all_best = []

    for fold_dir in fold_dirs:
        best_metrics = train_one_fold(
            fold_dir=fold_dir,
            project_root=project_root,
            output_root=output_root,
            device=device,
            model_name=args.model_name,
            weights_path=args.weights_path,
            image_size=args.image_size,
            emb_dim=args.emb_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
        )
        all_best.append(best_metrics)

    all_best_df = pd.DataFrame(all_best)
    all_best_df.to_csv(output_root / "cv_best_metrics.csv", index=False)

    summary = {
        "model_name": args.model_name,
        "num_folds": len(all_best),
        "mean_best_recall_at_1": float(all_best_df["best_val_recall_at_1"].mean()),
        "std_best_recall_at_1": float(all_best_df["best_val_recall_at_1"].std(ddof=0)),
        "mean_best_recall_at_5": float(all_best_df["best_val_recall_at_5"].mean()),
        "std_best_recall_at_5": float(all_best_df["best_val_recall_at_5"].std(ddof=0)),
    }

    with open(output_root / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


    print("Training summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()