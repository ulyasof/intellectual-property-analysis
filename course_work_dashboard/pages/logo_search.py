import os
import tempfile
import pandas as pd
import streamlit as st

from api.service import search_similar_logos
from utils.ui import apply_custom_styles, page_header, section_header

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
                suffix = os.path.splitext(uploaded_file.name)[1].lower()
                if not suffix:
                    suffix = ".jpg"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_file_path = tmp_file.name

                response = search_similar_logos(
                    file_name=temp_file_path,
                    top_k=top_k
                )

                nested_payload = response.get("results", {})
                if not isinstance(nested_payload, dict):
                    nested_payload = {}

                top_matches = nested_payload.get("results", [])
                if not isinstance(top_matches, list):
                    top_matches = []

                top_matches = top_matches[:top_k]

                risk = nested_payload.get("risk", {})
                if not isinstance(risk, dict):
                    risk = {}

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
                    query_mktu_classes = risk.get("query_mktu_classes", "—")
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
                        overlap_text = ", ".join(mktu_overlap) if mktu_overlap else "—"
                        st.write(f"**Классы МКТУ запроса:** {query_mktu_classes}")
                        st.write(f"**Классы МКТУ найденного знака:** {candidate_mktu_classes}")
                        st.write(f"**Пересечение по МКТУ:** {overlap_text}")

                    if explanation:
                        st.markdown("**Пояснение**")
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