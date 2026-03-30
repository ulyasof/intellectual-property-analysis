import streamlit as st


def apply_custom_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .page-subtitle {
            font-size: 1.05rem;
            color: #A9B4C7;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
        }

        .soft-card {
            background: #151E2E;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }

        .info-card {
            background: linear-gradient(135deg, #151E2E 0%, #1B2740 100%);
            border: 1px solid rgba(79,139,249,0.25);
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }

        .small-muted {
            color: #A9B4C7;
            font-size: 0.95rem;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.2rem;
        }

        .badge-high {
            background: rgba(255, 78, 78, 0.16);
            color: #FF8A8A;
            border: 1px solid rgba(255, 78, 78, 0.32);
        }

        .badge-medium {
            background: rgba(255, 184, 0, 0.14);
            color: #FFD166;
            border: 1px solid rgba(255, 184, 0, 0.28);
        }

        .badge-low {
            background: rgba(65, 201, 123, 0.14);
            color: #7FE6A8;
            border: 1px solid rgba(65, 201, 123, 0.28);
        }

        div[data-testid="stMetric"] {
            background: #151E2E;
            border: 1px solid rgba(255,255,255,0.08);
            padding: 14px;
            border-radius: 16px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
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
            <div style="font-weight:700; margin-bottom:6px;">{title}</div>
            <div class="small-muted">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def soft_card(title: str, text: str):
    st.markdown(
        f"""
        <div class="soft-card">
            <div style="font-weight:700; margin-bottom:6px;">{title}</div>
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