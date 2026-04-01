import pandas as pd
import plotly.express as px
import streamlit as st

from utils.columns import COLUMN_LABELS

from api.service import (
    get_cluster_stats,
    get_numeric_summary,
)
from utils.ui import apply_custom_styles, info_card, page_header, section_header

st.set_page_config(page_title="Главная", layout="wide")
apply_custom_styles()

page_header(
    "Аналитический дашборд по товарным знакам",
    "Дашборд объединяет обзор по компаниям, карточку компании с поиском по ИНН, кластерный анализ и сценарий поиска похожих товарных знаков."
)
summary = get_numeric_summary()
cluster_stats = get_cluster_stats()
cluster_df = pd.DataFrame(cluster_stats)

available_columns = list(COLUMN_LABELS.keys())


cluster_count = cluster_df["cluster"].nunique() if not cluster_df.empty else 0

k1, k2, k3 = st.columns(3)
k1.metric("Всего компаний", summary["row_count"])
k2.metric("Количество кластеров", cluster_count)
k3.metric("Количество признаков", len(available_columns))

section_header("О системе")
c1, c2 = st.columns(2)

with c1:
    info_card(
        "Что реализовано",
        "Реализованы список компаний с фильтрацией, карточка компании с поиском по введённому ИНН, визуализация кластеров компаний и mock-сценарий поиска похожих товарных знаков по изображению."
    )

with c2:
    info_card(
        "Зачем нужен дашборд",
        "Дашборд выступает единым пользовательским интерфейсом для анализа портфелей товарных знаков, сегментации компаний и просмотра ключевых характеристик правообладателей."
    )


section_header("Интерпретация кластеров")

st.caption("Кластеры отражают разные стратегии управления портфелем товарных знаков.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Минимальный портфель")
    st.markdown("- 1 активный товарный знак")
    st.markdown("- молодой портфель")
    st.markdown("- слабая бренд-стратегия")

with col2:
    st.markdown("#### Средний портфель")
    st.markdown("- несколько товарных знаков")
    st.markdown("- более системное управление")
    st.markdown("- устойчивая бренд-стратегия")

with col3:
    st.markdown("#### Развитый портфель")
    st.markdown("- крупный портфель знаков")
    st.markdown("- более зрелая структура")
    st.markdown("- бренды как инструмент роста")

st.success(
    "Ключевой вывод: более развитый портфель товарных знаков обычно связан с большим масштабом бизнеса и более высокими темпами роста выручки."
)


section_header("Распределение компаний по кластерам")
if cluster_df.empty:
    st.warning("Нет данных по кластерам.")
else:
    cluster_df["cluster_str"] = cluster_df["cluster"].astype(str)

    fig = px.bar(
        cluster_df,
        x="cluster_str",
        y="count",
        labels={"cluster_str": "Кластер", "count": "Количество компаний"},
        title="Количество компаний в каждом кластере",
    )
    st.plotly_chart(fig, use_container_width=True)

section_header("Краткая сводка")
summary_df = pd.DataFrame(
    [
        {"Показатель": "Число строк", "Значение": summary["row_count"]},
        {"Показатель": "Число колонок", "Значение": len(available_columns)},
    ]
)
st.dataframe(summary_df, use_container_width=True, hide_index=True)


with st.expander("Показать список доступных колонок"):
    for col in available_columns:
        st.write(f"{COLUMN_LABELS[col]}")