import pandas as pd
import streamlit as st

from api.service import get_companies_short, get_company_by_inn
from utils.ui import apply_custom_styles, page_header, section_header, info_card

st.set_page_config(page_title="Карточка компании", layout="wide")
apply_custom_styles()

page_header(
    "Карточка компании",
    "Страница показывает подробную информацию о компании и ключевые характеристики её портфеля товарных знаков. Поиск карточки осуществляется по введённому ИНН."
)

companies_response = get_companies_short(limit=1000, offset=0)
companies_items = companies_response["items"]

if not companies_items:
    st.warning("Нет компаний для выбора.")
    st.stop()

companies_df = pd.DataFrame(companies_items)

st.info("Поиск карточки компании осуществляется по введённому ИНН.")

available_inns = companies_df["inn"].astype(str).tolist()

selected_inn = st.text_input(
    "Введите ИНН компании",
    value=available_inns[0],
    placeholder="Например, 7701234567"
).strip()

if not selected_inn:
    st.warning("Введите ИНН компании.")
    st.stop()

if selected_inn not in available_inns:
    st.error("Компания с таким ИНН не найдена в текущем наборе данных.")
    st.stop()


company = get_company_by_inn(selected_inn)

if not company:
    st.error("Компания не найдена.")
    st.stop()

company_name = company.get("company_name", "Название не указано")
st.caption(f"Найдена компания: {company_name}")


if company is None:
    st.error("Не удалось загрузить карточку компании.")
    st.stop()

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
#k4.metric("Число классов МКТУ", company["nice_class_count"])

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
    ]
)
st.dataframe(company_table, use_container_width=True, hide_index=True)