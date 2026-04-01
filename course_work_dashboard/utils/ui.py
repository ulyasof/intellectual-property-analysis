import streamlit as st


def apply_custom_styles():
    st.markdown(
        """
        <style>
        .main-title {
            color: var(--text-color, #F3F6FB);
        }

        .page-subtitle {
            color: var(--text-color, #A9B4C7);
            opacity: 0.8;
        }

        .section-title {
            color: var(--text-color, #F3F6FB);
        }

        .soft-card {
            background: var(--secondary-background-color, #151E2E);
            border: 1px solid rgba(127, 127, 127, 0.18);
            color: var(--text-color, #F3F6FB);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }

        .soft-card-title {
            color: var(--text-color, #F3F6FB);
            font-weight: 700;
            margin-bottom: 6px;
        }

        .info-card {
            background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%);
            border: 1px solid rgba(37, 99, 235, 0.35);
            color: #FFFFFF;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }

        .info-card-title {
            color: #FFFFFF;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .small-muted {
            color: var(--text-color, #A9B4C7);
            opacity: 0.8;
            font-size: 0.95rem;
        }

        .info-card .small-muted {
            color: rgba(255, 255, 255, 0.88);
            opacity: 1;
        }

        div[data-testid="stMetric"] {
            background: var(--secondary-background-color, #151E2E);
            border: 1px solid rgba(127, 127, 127, 0.18);
            border-radius: 16px;
            padding: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str):
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def info_card(title: str, text: str):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{title}</div>
            <div class="small-muted">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def soft_card(title: str, text: str):
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="soft-card-title">{title}</div>
            <div class="small-muted">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str):
    level = str(level).lower()

    if level == "high":
        badge_class = "badge badge-high"
        label = "Высокий риск"
    elif level == "medium":
        badge_class = "badge badge-medium"
        label = "Средний риск"
    else:
        badge_class = "badge badge-low"
        label = "Низкий риск"

    st.markdown(f'<span class="{badge_class}">{label}</span>', unsafe_allow_html=True)