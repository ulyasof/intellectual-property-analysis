from pathlib import Path
import argparse
import pandas as pd


def safe_get(row, key):
    return row[key] if key in row and pd.notna(row[key]) else ""


def main():
    parser = argparse.ArgumentParser(
        description="Build final retrieval training set from all eligible tm_id."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="prepared/images_metadata_final.csv",
        help="Путь к images_metadata_final.csv"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="prepared/final_retrieval",
        help="Папка, куда сохранить финальные retrieval CSV"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной CSV: {input_path}")

    df = pd.read_csv(input_path, dtype=str).fillna("")

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

    registry_tm_ids = set(registry_df["tm_id"].unique())
    real_tm_ids = set(real_df["tm_id"].unique())

    eligible_tm_ids = sorted(registry_tm_ids & real_tm_ids)

    if not eligible_tm_ids:
        raise ValueError("Не найдено ни одного eligible tm_id с registry + real.")

    gallery_registry = registry_df.copy()
    gallery_registry["has_real_match"] = gallery_registry["tm_id"].isin(eligible_tm_ids).astype(int)
    gallery_registry = gallery_registry.sort_values(["tm_id", "image_id"]).reset_index(drop=True)
    gallery_registry.to_csv(out_dir / "gallery_registry.csv", index=False)

    final_queries_real = real_df[real_df["tm_id"].isin(eligible_tm_ids)].copy()
    final_queries_real = final_queries_real.sort_values(["tm_id", "image_id"]).reset_index(drop=True)
    final_queries_real.to_csv(out_dir / "final_queries_real.csv", index=False)

    pair_rows = []
    pair_id = 1

    for tm_id in eligible_tm_ids:
        tm_real = real_df[real_df["tm_id"] == tm_id]
        tm_registry = registry_df[registry_df["tm_id"] == tm_id]

        for _, q in tm_real.iterrows():
            for _, t in tm_registry.iterrows():
                pair_rows.append(
                    {
                        "pair_id": pair_id,
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

    final_pairs = pd.DataFrame(pair_rows)
    if final_pairs.empty:
        raise ValueError("Не удалось собрать final_pairs.csv")

    final_pairs = final_pairs.sort_values(["tm_id", "query_image_id", "target_image_id"]).reset_index(drop=True)
    final_pairs.to_csv(out_dir / "final_pairs.csv", index=False)

    pd.DataFrame({"tm_id": eligible_tm_ids}).to_csv(out_dir / "eligible_tm_ids.csv", index=False)

    summary = pd.DataFrame([{
        "num_registry_images": len(registry_df),
        "num_real_images": len(real_df),
        "num_eligible_tm_ids": len(eligible_tm_ids),
        "num_final_queries_real": len(final_queries_real),
        "num_final_pairs": len(final_pairs),
        "num_gallery_registry": len(gallery_registry),
    }])
    summary.to_csv(out_dir / "final_retrieval_summary.csv", index=False)


    print("Final retrieval set built")
    print(f"Input:  {input_path}")
    print(f"Output: {out_dir}")
    print()
    print(f"Eligible tm_id:        {len(eligible_tm_ids)}")
    print(f"Gallery registry rows: {len(gallery_registry)}")
    print(f"Final queries rows:    {len(final_queries_real)}")
    print(f"Final pairs rows:      {len(final_pairs)}")
    print()
    print("Saved files:")
    print(f" - {out_dir / 'gallery_registry.csv'}")
    print(f" - {out_dir / 'final_queries_real.csv'}")
    print(f" - {out_dir / 'final_pairs.csv'}")
    print(f" - {out_dir / 'eligible_tm_ids.csv'}")
    print(f" - {out_dir / 'final_retrieval_summary.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()