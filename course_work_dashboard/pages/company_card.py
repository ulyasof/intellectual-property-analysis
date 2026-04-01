import pandas as pd
import streamlit as st

from api.service import (
    get_company_by_inn,
    get_company_trademark_classes,
    get_company_trademark_classes_agg,
)
from utils.ui import apply_custom_styles, page_header, section_header, info_card

st.set_page_config(page_title="Карточка компании", layout="wide")
apply_custom_styles()

page_header(
    "Карточка компании",
    "Страница показывает подробную информацию о компании и ключевые характеристики её портфеля товарных знаков. Поиск карточки осуществляется по введённому ИНН."
)

st.info("Поиск карточки компании осуществляется по введенному ИНН.")

selected_inn = st.text_input(
    "Введите ИНН компании",
    placeholder="Например, 7701234567"
).strip()

if not selected_inn:
    st.warning("Введите ИНН компании.")
    st.stop()

company = get_company_by_inn(selected_inn)

if not company:
    st.error("Компания с таким ИНН не найдена.")
    st.stop()

company_classes = get_company_trademark_classes(selected_inn)
company_classes_agg = get_company_trademark_classes_agg(selected_inn)

class_items = company_classes.get("items", [])
unique_classes = company_classes_agg.get("unique_classes", [])
classes_count_map = company_classes_agg.get("classes_count", {})

company_name = company.get("company_name", "Название не указано")
st.caption(f"Найдена компания: {company_name}")

section_header("Общая информация")
col1, col2 = st.columns(2)

with col1:
    info_card(
        "Основные сведения",
        f"Название: {company['company_name']}<br>ИНН: {company['inn']}<br>Регион: {company['region']}"
    )

with col2:
    info_card(
        "Позиция в структуре данных",
        f"Отрасль: {company['industry']}<br>Кластер: {company['cluster']}"
    )

section_header("Ключевые показатели компании")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Товарных знаков", company.get("num_marks", "—"))
k2.metric("Доля активных ТЗ", f"{company.get('active_share', 0) * 100:.1f}%")
k3.metric("Средний возраст портфеля", round(company.get("avg_age", 0), 1))
k4.metric("Уникальных классов МКТУ", len(unique_classes))

section_header("Характеристики компании")
company_table = pd.DataFrame(
    [
        {"Параметр": "Название компании", "Значение": company["company_name"]},
        {"Параметр": "ИНН", "Значение": company["inn"]},
        {"Параметр": "Регион", "Значение": company["region"]},
        {"Параметр": "Отрасль", "Значение": company["industry"]},
        {"Параметр": "Количество товарных знаков", "Значение": company["num_marks"]},
        {"Параметр": "Доля активных знаков", "Значение": f"{company['active_share'] * 100:.1f}%"},
        {"Параметр": "Средний возраст портфеля", "Значение": f"{company['avg_age']:.1f} лет"},
        {"Параметр": "Кластер", "Значение": company["cluster"]},
        {
            "Параметр": "Уникальные классы МКТУ",
            "Значение": ", ".join(str(x) for x in unique_classes) if unique_classes else "—"
        },
    ]
)
st.dataframe(company_table, use_container_width=True, hide_index=True)

section_header("Классы МКТУ компании")

tab1, tab2 = st.tabs(["По товарным знакам", "Агрегация по классам"])

with tab1:
    if not class_items:
        st.info("По этой компании нет данных по классам МКТУ.")
    else:
        per_mark_df = pd.DataFrame(class_items)

        if "classes_list" in per_mark_df.columns:
            per_mark_df["classes_list"] = per_mark_df["classes_list"].apply(
                lambda x: ", ".join(str(v) for v in x) if isinstance(x, list) else x
            )

        per_mark_df = per_mark_df[
            ["reg_num", "classes", "classes_list", "classes_count"]
        ].rename(
            columns={
                "reg_num": "Номер регистрации",
                "classes": "Классы МКТУ",
                "classes_list": "Список классов МКТУ",
                "classes_count": "Количество классов",
            }
        )

        st.caption("Для каждого товарного знака показаны его классы МКТУ.")
        st.dataframe(per_mark_df, use_container_width=True, hide_index=True)

with tab2:
    if not classes_count_map:
        st.info("Нет агрегированных данных по классам МКТУ.")
    else:
        agg_df = pd.DataFrame(
            [
                {"Класс МКТУ": int(cls), "Количество вхождений": count}
                for cls, count in classes_count_map.items()
            ]
        ).sort_values(["Количество вхождений", "Класс МКТУ"], ascending=[False, True])

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Уникальных классов", len(unique_classes))

        with c2:
            st.metric("Всего вхождений классов", int(agg_df["Количество вхождений"].sum()))

        st.caption("Показано, сколько раз каждый класс МКТУ встречается в портфеле компании.")
        st.dataframe(agg_df, use_container_width=True, hide_index=True)

        st.bar_chart(
            agg_df.set_index("Класс МКТУ")["Количество вхождений"],
            use_container_width=True,
        )