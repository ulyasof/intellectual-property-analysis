import math
import pandas as pd
import streamlit as st

from api.service import filter_companies, get_cluster_stats

SHOW_EXPERIMENTAL_BRAND_SCORE = False

st.set_page_config(page_title="Компании", layout="wide")

st.title("Компании")
st.write(
    "На этой странице представлен список компаний с фильтрацией и пагинацией. "
    "Пользователь может отобрать компании по кластеру и просмотреть основные характеристики портфеля товарных знаков."
)

if "companies_offset" not in st.session_state:
    st.session_state["companies_offset"] = 0

if "companies_filters_signature" not in st.session_state:
    st.session_state["companies_filters_signature"] = None

limit = st.selectbox(
    "Сколько компаний показывать на странице",
    options=[5, 10, 20],
    index=0
)

cluster_stats = get_cluster_stats()
cluster_options = ["Все"] + [str(item["cluster"]) for item in cluster_stats]

selected_cluster = st.selectbox(
    "Фильтр по кластеру",
    options=cluster_options
)

min_brand_score = None
max_brand_score = None

if SHOW_EXPERIMENTAL_BRAND_SCORE:
    with st.expander("Экспериментальные фильтры brand_score"):
        min_brand_score = st.slider("Минимальный brand_score", 0.0, 1.0, 0.0, 0.01)
        max_brand_score = st.slider("Максимальный brand_score", 0.0, 1.0, 1.0, 0.01)

        if min_brand_score > max_brand_score:
            st.error("Минимальный brand_score не может быть больше максимального.")
            st.stop()

cluster_value = None if selected_cluster == "Все" else int(selected_cluster)

# Сбрасываем пагинацию, если поменялись фильтры или размер страницы
current_filters_signature = (
    limit,
    selected_cluster,
    min_brand_score,
    max_brand_score,
)

if st.session_state["companies_filters_signature"] != current_filters_signature:
    st.session_state["companies_filters_signature"] = current_filters_signature
    st.session_state["companies_offset"] = 0

response = filter_companies(
    cluster=cluster_value,
    min_brand_score=min_brand_score,
    max_brand_score=max_brand_score,
    limit=limit,
    offset=st.session_state["companies_offset"],
)

items = response["items"]
total = response["total"]
offset = response["offset"]

# если offset оказался за пределами total
if total > 0 and offset >= total:
    st.session_state["companies_offset"] = max(0, ((total - 1) // limit) * limit)
    st.rerun()

df = pd.DataFrame(items)

current_page = (st.session_state["companies_offset"] // limit) + 1
total_pages = max(1, math.ceil(total / limit))

m1, m2, m3 = st.columns(3)
m1.metric("Всего компаний после фильтрации", total)
m2.metric("Текущая страница", f"{current_page} из {total_pages}")
m3.metric("Записей на странице", len(df))

prev_col, next_col = st.columns(2)

with prev_col:
    if st.button("← Предыдущая страница", disabled=(st.session_state["companies_offset"] == 0)):
        st.session_state["companies_offset"] = max(0, st.session_state["companies_offset"] - limit)
        st.rerun()

with next_col:
    if st.button("Следующая страница →", disabled=(st.session_state["companies_offset"] + limit >= total)):
        st.session_state["companies_offset"] = st.session_state["companies_offset"] + limit
        st.rerun()

st.subheader("Таблица компаний")

if df.empty:
    st.warning("Нет компаний, подходящих под выбранные фильтры.")
else:
    column_labels = {
        "inn": "ИНН",
        "company_name": "Компания",
        "region": "Регион",
        "industry": "Отрасль",
        "num_marks": "Количество товарных знаков",
        "cluster": "Кластер",
        "brand_score": "Brand score",
        "active_share": "Доля активных ТЗ",
        "avg_portfolio_age": "Средний возраст портфеля",
        "nice_class_count": "Количество классов МКТУ",
        "pca_x": "PCA X",
        "pca_y": "PCA Y",
    }

    default_columns = [
        "inn",
        "company_name",
        "region",
        "industry",
        "num_marks",
        "cluster",
    ]

    available_columns = [
        "inn",
        "company_name",
        "region",
        "industry",
        "num_marks",
        "cluster",
        "brand_score",
        "active_share",
        "avg_portfolio_age",
        "nice_class_count",
    ]
    available_columns = [col for col in available_columns if col in df.columns]

    st.caption("Пользователь может самостоятельно выбрать, какие столбцы отображать в таблице.")

    selected_columns = st.multiselect(
        "Выберите столбцы для отображения",
        options=available_columns,
        default=[col for col in default_columns if col in available_columns],
        format_func=lambda col: column_labels.get(col, col),
    )

    if not selected_columns:
        st.info("Выберите хотя бы один столбец для отображения таблицы.")
    else:
        display_df = df[selected_columns].rename(
            columns={col: column_labels.get(col, col) for col in selected_columns}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)