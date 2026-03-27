from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_image_path(project_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return project_root / p


def load_image_rgb_on_white(image_path: Path) -> Image.Image:
    img = Image.open(image_path)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img).convert("RGB")
    else:
        img = img.convert("RGB")

    return img


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

# аугментации только для train
def build_train_transform(image_size: int = 224):
    return transforms.Compose([
        SquarePad(fill=(255, 255, 255)),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.05,
                hue=0.02
            )
        ], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
        ], p=0.2),
        transforms.RandomAffine(
            degrees=7,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            shear=3,
            fill=(255, 255, 255)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# для val/test 
def build_eval_transform(image_size: int = 224):
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


class PairDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        project_root: str | Path = ".",
        query_transform=None,
        target_transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.df = pd.read_csv(self.csv_path, dtype=str).fillna("")

        required_cols = {
            "tm_id",
            "label",
            "query_path",
            "target_path",
            "query_image_id",
            "target_image_id",
        }
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_path}: missing required columns: {missing}")

        self.query_transform = query_transform
        self.target_transform = target_transform if target_transform is not None else query_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        query_path = resolve_image_path(self.project_root, row["query_path"])
        target_path = resolve_image_path(self.project_root, row["target_path"])

        if not query_path.exists():
            raise FileNotFoundError(f"Query image not found: {query_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target image not found: {target_path}")

        query_img = load_image_rgb_on_white(query_path)
        target_img = load_image_rgb_on_white(target_path)

        if self.query_transform is not None:
            query_img = self.query_transform(query_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        return {
            "query_image": query_img,
            "target_image": target_img,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "tm_id": row["tm_id"],
            "query_path": row["query_path"],
            "target_path": row["target_path"],
            "query_image_id": row["query_image_id"],
            "target_image_id": row["target_image_id"],
        }


def build_datasets_for_fold(
    fold_dir: str | Path,
    project_root: str | Path = ".",
    image_size: int = 224,
):

    fold_dir = Path(fold_dir)

    train_csv = fold_dir / "train_pairs.csv"
    val_csv = fold_dir / "val_pairs.csv"

    train_ds = PairDataset(
        csv_path=train_csv,
        project_root=project_root,
        query_transform=build_train_transform(image_size=image_size),
        target_transform=build_train_transform(image_size=image_size),
    )

    val_ds = PairDataset(
        csv_path=val_csv,
        project_root=project_root,
        query_transform=build_eval_transform(image_size=image_size),
        target_transform=build_eval_transform(image_size=image_size),
    )

    return train_ds, val_ds