from pathlib import Path
import pandas as pd


RESULTS_ROOT = Path("checkpoints_metric_asym_lr1e6_seed42")

OUT_CSV = Path("prepared/hard_negatives_oof.csv")
MAX_NEG_PER_QUERY = 2


def get_predictions_csv(fold_dir: Path):
    for name in ["best_val_topk_predictions.csv", "best_val_top5_predictions.csv"]:
        p = fold_dir / name
        if p.exists():
            return p
    return None


def get_available_ranks(df: pd.DataFrame):
    ranks = []
    for r in range(1, 21):
        if f"top{r}_tm_id" in df.columns:
            ranks.append(r)
    return ranks


all_rows = []

fold_dirs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir() and p.name.startswith("fold_")])
if not fold_dirs:
    raise RuntimeError(f"Не найдены fold_* папки в {RESULTS_ROOT}")

for fold_dir in fold_dirs:
    csv_path = get_predictions_csv(fold_dir)
    if csv_path is None:
        print(f"Пропуск {fold_dir}: нет best_val_top5_predictions.csv")
        continue

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    ranks = get_available_ranks(df)

    for _, row in df.iterrows():
        q_tm_id = row["query_tm_id"]
        q_img_id = row["query_image_id"]
        q_path = row["query_path"]

        retrieved = []
        for r in ranks:
            retrieved.append({
                "rank": r,
                "tm_id": row.get(f"top{r}_tm_id", ""),
                "image_id": row.get(f"top{r}_image_id", ""),
                "path": row.get(f"top{r}_path", ""),
                "score": row.get(f"top{r}_score", ""),
            })

        first_correct_rank = None
        for item in retrieved:
            if item["tm_id"] == q_tm_id:
                first_correct_rank = item["rank"]
                break

        chosen_negatives = []

        if first_correct_rank is None:
            # правильного ответа нет в top-k значит берем первые неправильные
            for item in retrieved:
                if item["tm_id"] != q_tm_id and item["tm_id"] != "":
                    chosen_negatives.append(item)
                if len(chosen_negatives) >= MAX_NEG_PER_QUERY:
                    break

        elif first_correct_rank > 1:
            # правильный ответ есть, но не на 1 месте значит берем тех, кто стоит выше него
            for item in retrieved:
                if item["rank"] < first_correct_rank and item["tm_id"] != q_tm_id and item["tm_id"] != "":
                    chosen_negatives.append(item)
                if len(chosen_negatives) >= MAX_NEG_PER_QUERY:
                    break

        else:
            # top1 уже правильный значит hard negative не берем
            continue

        for neg in chosen_negatives:
            all_rows.append({
                "query_image_id": q_img_id,
                "query_tm_id": q_tm_id,
                "query_path": q_path,
                "negative_image_id": neg["image_id"],
                "negative_tm_id": neg["tm_id"],
                "negative_path": neg["path"],
                "rank": neg["rank"],
                "score": neg["score"],
                "fold": fold_dir.name,
            })

if not all_rows:
    raise RuntimeError("Hard negatives не собраны. Проверь prediction CSV.")

hard_df = pd.DataFrame(all_rows)
hard_df = hard_df.drop_duplicates(subset=["query_image_id", "negative_image_id"]).reset_index(drop=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
hard_df.to_csv(OUT_CSV, index=False)

print(f"Сохранено {len(hard_df)} hard negatives -> {OUT_CSV}")
print(hard_df.head(10).to_string(index=False))