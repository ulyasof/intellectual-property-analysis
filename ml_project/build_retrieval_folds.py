from pathlib import Path
import argparse
import random
import pandas as pd

# делим список items на n_folds частей
def make_folds(items, n_folds=5, seed=42):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    folds = [[] for _ in range(n_folds)]
    for i, item in enumerate(items):
        folds[i % n_folds].append(item)

    return folds


def safe_get(row, key):
    return row[key] if key in row and pd.notna(row[key]) else ""


# строит ositive pairs: query = real target = registry
def build_positive_pairs(real_subset, registry_subset, split_name):
    pair_rows = []
    pair_id = 1

    tm_ids = sorted(set(real_subset["tm_id"].unique()) & set(registry_subset["tm_id"].unique()))

    for tm_id in tm_ids:
        tm_real = real_subset[real_subset["tm_id"] == tm_id]
        tm_registry = registry_subset[registry_subset["tm_id"] == tm_id]

        for _, q in tm_real.iterrows():
            for _, t in tm_registry.iterrows():
                pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "retrieval_split": split_name,
                        "label": 1,
                        "tm_id": tm_id,

                        "query_image_id": safe_get(q, "image_id"),
                        "query_path": safe_get(q, "path"),
                        "query_source_type": safe_get(q, "source_type"),
                        "query_variant_status": safe_get(q, "variant_status"),
                        "query_brand_name": safe_get(q, "brand_name"),
                        "query_mktu_classes": safe_get(q, "mktu_classes"),
                        "query_status": safe_get(q, "status"),

                        "target_image_id": safe_get(t, "image_id"),
                        "target_path": safe_get(t, "path"),
                        "target_source_type": safe_get(t, "source_type"),
                        "target_variant_status": safe_get(t, "variant_status"),
                        "target_brand_name": safe_get(t, "brand_name"),
                        "target_mktu_classes": safe_get(t, "mktu_classes"),
                        "target_status": safe_get(t, "status"),
                    }
                )
                pair_id += 1

    return pd.DataFrame(pair_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build retrieval cross-validation folds for trademark search."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="prepared/images_metadata_final.csv",
        help="Путь к входному CSV"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="prepared/retrieval_cv",
        help="Папка, куда будут сохранены fold-ы"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Количество fold-ов"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной CSV: {input_path}")

    df = pd.read_csv(input_path, dtype=str)

    required_cols = {"image_id", "tm_id", "path", "source_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Во входном CSV не хватает колонок: {missing}")

    if "split" in df.columns:
        df = df.rename(columns={"split": "old_split"})

    df["tm_id"] = df["tm_id"].astype(str).str.strip()
    df["source_type"] = df["source_type"].astype(str).str.strip().str.lower()

    registry_df = df[df["source_type"] == "registry"].copy()
    real_df = df[df["source_type"] == "real"].copy()

    if registry_df.empty:
        raise ValueError("Во входных данных нет registry-изображений")
    if real_df.empty:
        raise ValueError("Во входных данных нет real-изображений")

    registry_tm_ids = set(registry_df["tm_id"].unique())
    real_tm_ids = set(real_df["tm_id"].unique())

    eligible_tm_ids = sorted(registry_tm_ids & real_tm_ids)

    if len(eligible_tm_ids) < args.n_folds:
        raise ValueError(
            f"Eligible tm_id = {len(eligible_tm_ids)}, а fold-ов = {args.n_folds}. "
            f"eligible_tm_id  не меньше числа fold-ов"
        )

    folds = make_folds(eligible_tm_ids, n_folds=args.n_folds, seed=args.seed)

    pd.DataFrame({"tm_id": eligible_tm_ids}).to_csv(out_dir / "eligible_tm_ids.csv", index=False)

    gallery_registry_all = registry_df.copy()
    gallery_registry_all["has_real_match"] = gallery_registry_all["tm_id"].isin(eligible_tm_ids).astype(int)
    gallery_registry_all = gallery_registry_all.sort_values(["tm_id", "image_id"]).reset_index(drop=True)


    print("Создание retrieval cross-validation folds")
    print()
    print(f"Input file: {input_path}")
    print(f"Output dir: {out_dir}")
    print()
    print(f"все registry images: {len(registry_df)}")
    print(f"все real images: {len(real_df)}")
    print(f"Eligible tm_id (registry + real): {len(eligible_tm_ids)}")
    print(f"количество folds: {args.n_folds}")
    print()

    for fold_idx in range(args.n_folds):
        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_tm_ids = sorted(folds[fold_idx])
        train_tm_ids = sorted([tm_id for i, fold in enumerate(folds) if i != fold_idx for tm_id in fold])

        train_tm_set = set(train_tm_ids)
        val_tm_set = set(val_tm_ids)

        train_real = real_df[real_df["tm_id"].isin(train_tm_set)].copy()
        val_real = real_df[real_df["tm_id"].isin(val_tm_set)].copy()

        train_registry = registry_df[registry_df["tm_id"].isin(train_tm_set)].copy()
        val_registry = registry_df[registry_df["tm_id"].isin(val_tm_set)].copy()

        train_real["retrieval_split"] = "train"
        val_real["retrieval_split"] = "val"

        train_registry["retrieval_split"] = "train"
        val_registry["retrieval_split"] = "val"

        train_pairs = build_positive_pairs(train_real, train_registry, split_name="train")
        val_pairs = build_positive_pairs(val_real, val_registry, split_name="val")

        gallery_registry = gallery_registry_all.copy()
        gallery_registry["fold_val_target"] = gallery_registry["tm_id"].isin(val_tm_set).astype(int)
        gallery_registry["fold_train_seen"] = gallery_registry["tm_id"].isin(train_tm_set).astype(int)

        train_real = train_real.sort_values(["tm_id", "image_id"]).reset_index(drop=True)
        val_real = val_real.sort_values(["tm_id", "image_id"]).reset_index(drop=True)
        train_pairs = train_pairs.sort_values(["tm_id", "query_image_id", "target_image_id"]).reset_index(drop=True)
        val_pairs = val_pairs.sort_values(["tm_id", "query_image_id", "target_image_id"]).reset_index(drop=True)

        gallery_registry.to_csv(fold_dir / "gallery_registry.csv", index=False)
        train_real.to_csv(fold_dir / "train_queries_real.csv", index=False)
        val_real.to_csv(fold_dir / "val_queries_real.csv", index=False)
        train_pairs.to_csv(fold_dir / "train_pairs.csv", index=False)
        val_pairs.to_csv(fold_dir / "val_pairs.csv", index=False)

        with open(fold_dir / "train_tm_ids.txt", "w", encoding="utf-8") as f:
            for tm_id in train_tm_ids:
                f.write(f"{tm_id}\n")

        with open(fold_dir / "val_tm_ids.txt", "w", encoding="utf-8") as f:
            for tm_id in val_tm_ids:
                f.write(f"{tm_id}\n")

        summary = {
            "fold": fold_idx,
            "train_tm_ids": len(train_tm_ids),
            "val_tm_ids": len(val_tm_ids),
            "train_real_rows": len(train_real),
            "val_real_rows": len(val_real),
            "train_pairs_rows": len(train_pairs),
            "val_pairs_rows": len(val_pairs),
            "gallery_registry_rows": len(gallery_registry),
        }
        pd.DataFrame([summary]).to_csv(fold_dir / "fold_summary.csv", index=False)

        print(f"[fold_{fold_idx}]")
        print(f"  train tm_id: {len(train_tm_ids)}")
        print(f"  val tm_id:   {len(val_tm_ids)}")
        print(f"  train real:  {len(train_real)}")
        print(f"  val real:    {len(val_real)}")
        print(f"  train pairs: {len(train_pairs)}")
        print(f"  val pairs:   {len(val_pairs)}")
        print(f"  gallery:     {len(gallery_registry)}")
        print()

    print("Created fold structure in:")
    print(out_dir)



if __name__ == "__main__":
    main()