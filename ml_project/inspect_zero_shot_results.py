from pathlib import Path
import math
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(".")
CV_DIR = Path("prepared/retrieval_cv")
MAX_ROWS_PER_FOLD = 30

CELL_W = 260
CELL_H = 260
TEXT_H = 60
BG = "white"
FONT_FILL = "black"


def open_img(rel_path: str):
    p = PROJECT_ROOT / rel_path
    img = Image.open(p).convert("RGB")
    img.thumbnail((CELL_W - 20, CELL_H - 20))
    canvas = Image.new("RGB", (CELL_W, CELL_H), BG)
    x = (CELL_W - img.width) // 2
    y = (CELL_H - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


fold_dirs = sorted([p for p in CV_DIR.iterdir() if p.is_dir() and p.name.startswith("fold_")])

for fold_dir in fold_dirs:
    csv_path = fold_dir / "zero_shot_top5_predictions.csv"
    out_dir = fold_dir / "visual_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Skip: {csv_path} not found")
        continue

    df = pd.read_csv(csv_path).fillna("")

    for idx, row in df.head(MAX_ROWS_PER_FOLD).iterrows():
        items = [
            ("QUERY", row["query_path"], f"tm_id={row['query_tm_id']}"),
            ("TOP1", row["top1_path"], f"tm_id={row['top1_tm_id']} score={row['top1_score']:.4f}"),
            ("TOP2", row["top2_path"], f"tm_id={row['top2_tm_id']} score={row['top2_score']:.4f}"),
            ("TOP3", row["top3_path"], f"tm_id={row['top3_tm_id']} score={row['top3_score']:.4f}"),
            ("TOP4", row["top4_path"], f"tm_id={row['top4_tm_id']} score={row['top4_score']:.4f}"),
            ("TOP5", row["top5_path"], f"tm_id={row['top5_tm_id']} score={row['top5_score']:.4f}"),
        ]

        cols = 3
        rows = math.ceil(len(items) / cols)
        canvas = Image.new("RGB", (cols * CELL_W, rows * (CELL_H + TEXT_H)), BG)
        draw = ImageDraw.Draw(canvas)

        for i, (title, rel_path, subtitle) in enumerate(items):
            c = i % cols
            r = i // cols
            x = c * CELL_W
            y = r * (CELL_H + TEXT_H)

            img = open_img(rel_path)
            canvas.paste(img, (x, y))

            draw.rectangle([x, y + CELL_H, x + CELL_W, y + CELL_H + TEXT_H], fill="white")
            draw.text((x + 10, y + CELL_H + 5), title, fill=FONT_FILL)
            draw.text((x + 10, y + CELL_H + 25), subtitle, fill=FONT_FILL)

        hit1 = int(row["hit_at_1"])
        hit5 = int(row["hit_at_5"])
        out_name = f"row_{idx:03d}_hit1_{hit1}_hit5_{hit5}.jpg"
        canvas.save(out_dir / out_name, quality=95)

    print(f"Saved visual checks for {fold_dir.name} -> {out_dir}")