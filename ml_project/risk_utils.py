from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def parse_mktu_classes(value: str) -> set[str]:
    if value is None:
        return set()
    tokens = re.findall(r"\d{1,2}", str(value))
    return {token.zfill(2) for token in tokens}


def format_mktu_classes(value: str) -> str:
    classes = sorted(parse_mktu_classes(value))
    return " ".join(classes) if classes else "не указаны"


def normalize_score(score: float, low: float = 0.45, high: float = 0.85) -> float:
    if score <= low:
        return 0.0
    if score >= high:
        return 1.0
    return (score - low) / (high - low)


def compute_mktu_overlap_factor(query_mktu: str, candidate_mktu: str) -> tuple[float, list[str]]:
    query_set = parse_mktu_classes(query_mktu)
    candidate_set = parse_mktu_classes(candidate_mktu)

    if not query_set or not candidate_set:
        return 0.35, []

    overlap = sorted(query_set & candidate_set)
    if not overlap:
        return 0.0, []

    union = query_set | candidate_set
    jaccard = len(overlap) / len(union) if union else 0.0
    factor = min(1.0, 0.70 + 0.30 * jaccard)
    return factor, overlap


def crowding_factor(scores: Iterable[float]) -> float:
    top_scores = list(scores)[:3]
    if not top_scores:
        return 0.0
    avg_score = sum(top_scores) / len(top_scores)
    return normalize_score(avg_score, low=0.40, high=0.80)


def aggregate_results_by_tm_id(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df.copy()

    grouped = (
        results_df.groupby("tm_id", as_index=False)
        .agg(
            best_score=("score", "max"),
            mean_score=("score", "mean"),
            image_count=("score", "size"),
            path=("path", "first"),
            image_id=("image_id", "first"),
            source_type=("source_type", "first"),
            brand_name=("brand_name", "first"),
            mktu_classes=("mktu_classes", "first"),
            status=("status", "first"),
        )
    )

    grouped["agg_score"] = 0.7 * grouped["best_score"] + 0.3 * grouped["mean_score"]
    grouped = grouped.sort_values(["agg_score", "best_score"], ascending=False).reset_index(drop=True)
    return grouped


def estimate_trademark_risk(results_df: pd.DataFrame, query_mktu: str = "") -> dict:
    if results_df.empty:
        return {
            "risk_score": 0.0,
            "risk_level": "low",
            "top_conflict_tm_id": "",
            "top_visual_score": 0.0,
            "query_mktu_classes": format_mktu_classes(query_mktu),
            "candidate_mktu_classes": "",
            "mktu_overlap": [],
            "explanation": ["Похожие товарные знаки не найдены."],
        }

    unique_tm_df = aggregate_results_by_tm_id(results_df)
    top = unique_tm_df.iloc[0]

    visual_factor = normalize_score(float(top["agg_score"]))
    mktu_factor, overlap = compute_mktu_overlap_factor(query_mktu, top.get("mktu_classes", ""))
    crowd_factor_value = crowding_factor(unique_tm_df["agg_score"].tolist())

    risk_score = 100.0 * (
        0.65 * visual_factor +
        0.25 * mktu_factor +
        0.10 * crowd_factor_value
    )

    if visual_factor >= 0.90 and overlap:
        risk_score = max(risk_score, 85.0)
    elif visual_factor >= 0.80 and overlap:
        risk_score = max(risk_score, 70.0)

    risk_score = round(max(0.0, min(100.0, risk_score)), 1)

    if risk_score >= 75:
        risk_level = "high"
    elif risk_score >= 45:
        risk_level = "medium"
    else:
        risk_level = "low"

    explanation = [
        f"Самый конфликтный найденный знак: tm_id={top['tm_id']}.",
        f"Агрегированный visual score={float(top['agg_score']):.4f}.",
        f"Классы МКТУ запроса: {format_mktu_classes(query_mktu)}.",
        f"Классы МКТУ найденного знака: {format_mktu_classes(top.get('mktu_classes', ''))}.",
    ]

    if overlap:
        explanation.append(f"Пересечение по МКТУ: {', '.join(overlap)}.")
    else:
        explanation.append("Пересечение по МКТУ не найдено или классы не были указаны.")

    explanation.append(f"Фактор насыщенности похожими знаками: {crowd_factor_value:.2f}.")

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "top_conflict_tm_id": str(top["tm_id"]),
        "top_visual_score": round(float(top["agg_score"]), 4),
        "query_mktu_classes": format_mktu_classes(query_mktu),
        "candidate_mktu_classes": format_mktu_classes(top.get("mktu_classes", "")),
        "mktu_overlap": overlap,
        "top_brand_name": str(top.get("brand_name", "")),
        "num_unique_candidates": int(len(unique_tm_df)),
        "explanation": explanation,
    }