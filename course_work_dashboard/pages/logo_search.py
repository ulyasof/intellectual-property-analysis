import os
import tempfile
import pandas as pd
import streamlit as st

from api.service import search_similar_logos
from utils.ui import apply_custom_styles, page_header, section_header


def parse_mktu_classes(raw_value: str) -> list[int]:
    if not raw_value or not raw_value.strip():
        return []

    result = []
    for chunk in raw_value.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.isdigit():
            raise ValueError(
                "Классы МКТУ должны быть перечислены числами через запятую, например: 10, 27, 32"
            )
        result.append(int(chunk))

    return sorted(set(result))


def format_class_list(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "—"
    return str(value)


st.set_page_config(page_title="Поиск похожих знаков", layout="wide")
apply_custom_styles()

page_header(
    "Поиск похожих товарных знаков",
    "Страница реализует сценарий visual search: пользователь загружает изображение, а система возвращает наиболее близкие зарегистрированные обозначения."
)

uploaded_file = st.file_uploader(
    "Загрузите изображение логотипа",
    type=["png", "jpg", "jpeg", "webp"]
)

query_mktu_input = st.text_input(
    "Классы МКТУ для загруженного изображения",
    placeholder="Например: 10, 27, 32"
)

top_k = st.selectbox(
    "Количество результатов",
    options=[3, 5],
    index=1
)

if uploaded_file is not None:
    section_header("Загруженное изображение")
    st.image(uploaded_file, caption=uploaded_file.name, width=320)

    if st.button("Найти похожие знаки", use_container_width=True):
        with st.spinner("Выполняется поиск похожих знаков..."):
            temp_file_path = None

            try:
                query_mktu_classes = parse_mktu_classes(query_mktu_input)

                suffix = os.path.splitext(uploaded_file.name)[1].lower()
                if not suffix:
                    suffix = ".jpg"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_file_path = tmp_file.name

                response = search_similar_logos(
                    file_name=temp_file_path,
                    top_k=top_k,
                    query_mktu_classes=query_mktu_classes,
                )

                payload = response.get("results", {})
                top_matches = payload.get("results", [])[:top_k]
                risk = payload.get("risk", {})

                section_header("Результаты поиска")

                if not top_matches:
                    st.warning("Похожие знаки не найдены.")
                else:
                    st.success(f"Найдено совпадений: {len(top_matches)}")

                    summary_df = pd.DataFrame([
                        {
                            "Ранг": item.get("rank", "—"),
                            "Trademark ID": item.get("trademark_id", "—"),
                            "Similarity": round(item.get("similarity", 0), 4)
                            if item.get("similarity") is not None else "—",
                        }
                        for item in top_matches
                    ])

                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    st.markdown("### Найденные похожие знаки")

                    for item in top_matches:
                        image_url = item.get("image_url")
                        sim = item.get("similarity")

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            if image_url:
                                st.image(
                                    image_url,
                                    caption=f"TM {item.get('trademark_id', '—')}",
                                    use_container_width=True
                                )
                            else:
                                st.info("Изображение недоступно")

                        with col2:
                            st.markdown(f"#### Совпадение #{item.get('rank', '—')}")
                            st.write(f"**Trademark ID:** {item.get('trademark_id', '—')}")
                            st.write(
                                f"**Similarity:** {round(sim, 4) if sim is not None else '—'}"
                            )

                        st.divider()

                st.subheader("Оценка риска")

                if not risk:
                    st.info("Блок оценки риска не был возвращён микросервисом.")
                else:
                    risk_level = str(risk.get("risk_level", "")).lower()
                    risk_score = risk.get("risk_score", "—")
                    top_conflict_tm_id = risk.get("top_conflict_tm_id", "—")
                    top_visual_score = risk.get("top_visual_score", "—")
                    query_mktu_classes_from_risk = risk.get("query_mktu_classes", query_mktu_classes)
                    candidate_mktu_classes = risk.get("candidate_mktu_classes", "—")
                    mktu_overlap = risk.get("mktu_overlap", [])
                    num_unique_candidates = risk.get("num_unique_candidates", "—")
                    explanation = risk.get("explanation", [])

                    if risk_level == "low":
                        st.success(f"Уровень риска: LOW | Risk score: {risk_score}")
                    elif risk_level == "medium":
                        st.warning(f"Уровень риска: MEDIUM | Risk score: {risk_score}")
                    elif risk_level == "high":
                        st.error(f"Уровень риска: HIGH | Risk score: {risk_score}")
                    else:
                        st.info(
                            f"Уровень риска: {risk.get('risk_level', '—')} | Risk score: {risk_score}"
                        )

                    r1, r2 = st.columns(2)

                    with r1:
                        st.write(f"**Самый конфликтный знак:** {top_conflict_tm_id}")
                        st.write(f"**Top visual score:** {top_visual_score}")
                        st.write(f"**Количество уникальных кандидатов:** {num_unique_candidates}")

                    with r2:
                        st.write(
                            f"**Классы МКТУ запроса:** {format_class_list(query_mktu_classes_from_risk)}"
                        )
                        st.write(
                            f"**Классы МКТУ найденного знака:** {format_class_list(candidate_mktu_classes)}"
                        )
                        st.write(
                            f"**Пересечение по МКТУ:** {format_class_list(mktu_overlap)}"
                        )

                    if explanation:
                        st.markdown("**Пояснение**")
                        if isinstance(explanation, str):
                            st.markdown(f"- {explanation}")
                        else:
                            for line in explanation:
                                st.markdown(f"- {line}")

            except Exception as e:
                st.error(f"Ошибка при поиске похожих знаков: {e}")

            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

else:
    st.info(
        "Загрузите изображение PNG/JPG/JPEG/WEBP, чтобы выполнить поиск похожих товарных знаков."
    )