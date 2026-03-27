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
    PairDataset,
    build_train_transform,
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
            raise ValueError("Unsupported model_name")

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

        pos_logits_q = logits.masked_fill(~positive_mask, very_neg)
        log_pos_q = torch.logsumexp(pos_logits_q, dim=1)
        log_all_q = torch.logsumexp(logits, dim=1)
        loss_q = -(log_pos_q - log_all_q).mean()

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

        bs = images.shape[0]
        for i in range(bs):
            all_rows.append({
                "image_id": batch["image_id"][i],
                "tm_id": batch["tm_id"][i],
                "path": batch["path"][i],
                "source_type": batch["source_type"][i],
            })

    return torch.cat(all_embs, dim=0), pd.DataFrame(all_rows)


@torch.no_grad()
def evaluate_retrieval(model, final_dir: Path, project_root: Path, device, batch_size: int, num_workers: int, image_size: int):
    transform = build_eval_transform(image_size=image_size)

    queries_csv = final_dir / "final_queries_real.csv"
    gallery_csv = final_dir / "gallery_registry.csv"

    query_ds = SingleImageCsvDataset(queries_csv, project_root=project_root, transform=transform)
    gallery_ds = SingleImageCsvDataset(gallery_csv, project_root=project_root, transform=transform)

    query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    gallery_loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

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
    metrics = {
        "num_queries": num_queries,
        "gallery_size": len(gallery_meta),
        "recall_at_1": recall_at_1_hits / num_queries if num_queries else 0.0,
        "recall_at_5": recall_at_5_hits / num_queries if num_queries else 0.0,
    }

    return metrics, pd.DataFrame(top5_rows), gallery_embs, gallery_meta


def main():
    parser = argparse.ArgumentParser(description="Train final retrieval model on all eligible tm_id.")
    parser.add_argument("--final_dir", type=str, default="prepared/final_retrieval")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="final_model")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet18", "resnet50", "efficientnet_b0"])
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_backbone", type=float, default=3e-6)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)

    final_pairs_csv = final_dir / "final_pairs.csv"
    if not final_pairs_csv.exists():
        raise FileNotFoundError(f"Не найден {final_pairs_csv}. Сначала запусти build_final_retrieval_set.py")

    train_ds = PairDataset(
        csv_path=final_pairs_csv,
        project_root=project_root,
        query_transform=build_train_transform(image_size=args.image_size),
        target_transform=build_train_transform(image_size=args.image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = EmbeddingModel(
        model_name=args.model_name,
        emb_dim=args.emb_dim,
        weights_path=args.weights_path,
    ).to(device)

    criterion = MultiPositiveCrossModalLoss(temperature=args.temperature)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    history = []
    best_score = -1.0
    best_epoch = -1


    print("Train final retrieval model")
    print()
    print(f"Final dir:    {final_dir}")
    print(f"Project root: {project_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Model:        {args.model_name}")
    print(f"Device:       {device}")


    for epoch in range(1, args.epochs + 1):
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

        # не для отчета
        sanity_metrics, top5_df, gallery_embs, gallery_meta = evaluate_retrieval(
            model=model,
            final_dir=final_dir,
            project_root=project_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "sanity_recall_at_1": sanity_metrics["recall_at_1"],
            "sanity_recall_at_5": sanity_metrics["recall_at_5"],
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"sanity_R@1={sanity_metrics['recall_at_1']:.4f} | "
            f"sanity_R@5={sanity_metrics['recall_at_5']:.4f}"
        )

        if sanity_metrics["recall_at_5"] > best_score:
            best_score = sanity_metrics["recall_at_5"]
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_name": args.model_name,
                "emb_dim": args.emb_dim,
                "image_size": args.image_size,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_sanity_recall_at_5": best_score,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            top5_df.to_csv(output_dir / "best_final_top5_predictions.csv", index=False)
            gallery_meta.to_csv(output_dir / "gallery_metadata.csv", index=False)
            torch.save(gallery_embs, output_dir / "gallery_embeddings.pt")

            with open(output_dir / "best_sanity_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": epoch,
                        "sanity_recall_at_1": sanity_metrics["recall_at_1"],
                        "sanity_recall_at_5": sanity_metrics["recall_at_5"],
                        "num_queries": sanity_metrics["num_queries"],
                        "gallery_size": sanity_metrics["gallery_size"],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    torch.save(
        {
            "epoch": args.epochs,
            "model_name": args.model_name,
            "emb_dim": args.emb_dim,
            "image_size": args.image_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_dir / "last_model.pt",
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    final_summary = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "best_epoch_for_demo": best_epoch,
        "best_sanity_recall_at_5": best_score,
        "note": "Эти метрики получены на обучающей выборке и не используются как  оценка",
    }

    with open(output_dir / "final_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)


    print("Final model training finished")
    print()
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()