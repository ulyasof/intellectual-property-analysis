import pandas as pd
import plotly.express as px
import streamlit as st

from api.service import filter_companies, get_cluster_stats, get_pca_data
from utils.ui import apply_custom_styles, page_header, section_header, info_card

st.set_page_config(page_title="Кластеры", layout="wide")
apply_custom_styles()

page_header(
    "Кластеры компаний",
    "На странице представлены статистика по кластерам, визуализация компаний в пространстве признаков и интерпретация типов бренд-стратегий."
)

cluster_stats = get_cluster_stats()
pca_data = get_pca_data()

stats_df = pd.DataFrame(cluster_stats)
points_df = pd.DataFrame(pca_data)

if stats_df.empty or points_df.empty:
    st.warning("Нет данных для кластерной визуализации.")
    st.stop()

points_df["cluster_str"] = points_df["cluster"].astype(str)

k1, k2, k3 = st.columns(3)
k1.metric("Всего компаний", len(points_df))
k2.metric("Количество кластеров", stats_df["cluster"].nunique())
k3.metric("Точек на графике", len(points_df))

cluster_options = ["Все"] + sorted(points_df["cluster_str"].unique().tolist())
selected_cluster = st.selectbox("Выберите кластер", options=cluster_options)


section_header("Описание кластеров")
info_card(
    "Смысл кластеризации",
    "Кластеризация выделяет три типа компаний по стратегии управления товарными знаками: компании с минимальным портфелем, компании со средним портфелем и компании с развитым портфелем брендов."
)

info_card(
    "Как интерпретировать группы",
    "Для компаний с минимальным портфелем обычно характерен 1 активный товарный знак и молодой портфель. Компании со средним портфелем, как правило, имеют несколько зарегистрированных обозначений и более системную бренд-стратегию. Компании с развитым портфелем обладают более крупным и зрелым набором товарных знаков и активнее используют бренды как инструмент развития бизнеса."
)


if selected_cluster == "Все":
    filtered_points_df = points_df.copy()
else:
    filtered_points_df = points_df[points_df["cluster_str"] == selected_cluster]

section_header("Визуализация компаний в пространстве признаков")

fig = px.scatter(
    filtered_points_df,
    x="pca_x",
    y="pca_y",
    color="cluster_str",
    hover_name="company_name",
    hover_data={"inn": True, "brand_score": True, "pca_x": False, "pca_y": False},
    title="Компании в пространстве признаков",
)
fig.update_layout(
    xaxis_title="Компонента 1",
    yaxis_title="Компонента 2",
    legend_title="Кластер",
)
st.plotly_chart(fig, use_container_width=True)

section_header("Сводка по кластерам")
stats_display = stats_df.rename(
    columns={"cluster": "Кластер", "count": "Количество компаний"}
)
st.dataframe(stats_display, use_container_width=True, hide_index=True)

if selected_cluster != "Все":
    st.caption(
        "Ниже показаны усредненные характеристики компаний выбранной группы. "
        "Интерпретацию кластера следует делать по его показателям, а не только по номеру."
    )
    section_header(f"Характеристики выбранного кластера: {selected_cluster}")

    cluster_response = filter_companies(
        cluster=int(selected_cluster),
        limit=100,
        offset=0,
    )

    cluster_companies_df = pd.DataFrame(cluster_response["items"])

    if cluster_companies_df.empty:
        st.warning("В выбранном кластере нет компаний.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            info_card(
                "Размер кластера",
                f"Количество компаний: {len(cluster_companies_df)}<br>"
                f"Среднее число товарных знаков: {round(cluster_companies_df['num_marks'].mean(), 2)}"
            )

        with c2:
            info_card(
                "Устойчивость портфеля",
                f"Средняя доля активных знаков: {cluster_companies_df['active_share'].mean() * 100:.1f}%"
            )

        companies_view = cluster_companies_df[
            ["company_name", "inn", "region", "industry", "num_marks"]
        ].rename(
            columns={
                "company_name": "Компания",
                "inn": "ИНН",
                "region": "Регион",
                "industry": "Отрасль",
                "num_marks": "Товарных знаков",
            }
        )
        st.dataframe(companies_view, use_container_width=True, hide_index=True)