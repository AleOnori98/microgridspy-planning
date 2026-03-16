from __future__ import annotations

from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="MicroGridsPy - Planning",
    layout="wide",
)


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
REPOSITORY_URL = "https://github.com/AleOnori98/microgridspy-planning"
RAMP_URL = "https://github.com/AleOnori98/RAMP-Streamlit"
PVGIS_URL = "https://github.com/AleOnori98/PVGIS-Streamlit-App"
LV_TOPOLOGY_URL = "https://github.com/AleOnori98/LV-Distribution-Topology-Streamlit"


def _asset(name: str) -> str:
    return str(ASSETS_DIR / name)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --mgpy-ink: #153243;
            --mgpy-muted: #5b6b73;
            --mgpy-accent: #1f7a8c;
            --mgpy-gold: #f4b942;
            --mgpy-surface: #f7fbfc;
            --mgpy-border: rgba(21, 50, 67, 0.10);
        }
        .hero-wrap {
            padding: 1.75rem 1.5rem;
            border-radius: 22px;
            background:
                radial-gradient(circle at top right, rgba(244, 185, 66, 0.28), transparent 30%),
                linear-gradient(135deg, #f7fbfc 0%, #e8f4f6 52%, #fefaf1 100%);
            border: 1px solid var(--mgpy-border);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            color: var(--mgpy-accent);
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .hero-title {
            color: var(--mgpy-ink);
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }
        .hero-body {
            color: var(--mgpy-muted);
            font-size: 1.05rem;
            line-height: 1.65;
            max-width: 1000px;
        }
        .section-note {
            color: var(--mgpy-muted);
            font-size: 0.98rem;
            line-height: 1.6;
            margin-bottom: 0.75rem;
        }
        .step-card {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: white;
            border: 1px solid var(--mgpy-border);
            min-height: 150px;
        }
        .step-num {
            color: var(--mgpy-accent);
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .step-title {
            color: var(--mgpy-ink);
            font-size: 1.08rem;
            font-weight: 700;
            margin: 0.25rem 0 0.45rem 0;
        }
        .step-copy {
            color: var(--mgpy-muted);
            font-size: 0.94rem;
            line-height: 1.55;
        }
        .resource-box {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: #fff;
            border: 1px solid var(--mgpy-border);
            height: 100%;
        }
        .resource-title {
            color: var(--mgpy-ink);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .resource-copy {
            color: var(--mgpy-muted);
            font-size: 0.93rem;
            line-height: 1.55;
            margin-bottom: 0;
        }
        .featured-card {
            display: block;
            height: 0.01rem;
            margin: 0;
            padding: 0;
            opacity: 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.featured-card) {
            border: 2px solid rgba(47, 128, 237, 0.95) !important;
            box-shadow: 0 8px 24px rgba(47, 128, 237, 0.10);
            background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
        }
        .featured-kicker {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: rgba(47, 128, 237, 0.12);
            color: #2f80ed;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
        }
        div[data-testid="stButton"]:has(button[kind="primary"]) button {
            background: linear-gradient(135deg, #2f80ed 0%, #1f7a8c 100%);
            border: 1px solid #2f80ed;
            color: white;
            font-weight: 700;
        }
        .card-link {
            color: var(--mgpy-accent);
            font-weight: 600;
            text-decoration: none;
        }
        .card-link:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_featured_planning_card() -> None:
    with st.container(border=True):
        st.markdown('<div class="featured-card"></div>', unsafe_allow_html=True)
        col_image, col_body = st.columns([1.15, 1.45], gap="large")
        with col_image:
            st.image(_asset("planning_tool_card.png"), width="stretch")
        with col_body:
            st.markdown('<div class="featured-kicker">Core planning engine</div>', unsafe_allow_html=True)
            st.markdown("### MicroGridsPy Planning")
            st.write(
                "Techno-economic optimization of mini-grid systems under deterministic or stochastic assumptions. "
                "Use it to size renewables, batteries, generators, and grid interaction with either a representative typical year "
                "or a multi-year dynamic formulation with capacity expansion."
            )
            st.markdown(f"[GitHub repository]({REPOSITORY_URL})")
            if st.button("Open Project Setup", type="primary", key="open_project_setup"):
                st.switch_page("pages/0_Project_Setup.py")


def _tool_card(
    *,
    title: str,
    description: str,
    image_name: str,
    badge: str,
    repo_url: str | None = None,
) -> None:
    with st.container(border=True):
        st.image(_asset(image_name), width="stretch")
        st.caption(badge)
        st.markdown(f"**{title}**")
        st.write(description)
        if repo_url:
            st.markdown(f"[GitHub repository]({repo_url})")


def _render_ecosystem() -> None:
    st.title("Welcome to MicroGridsPy!")
    st.markdown(
        "**MicroGridsPy Planning** is the techno-economic optimization layer of the MicroGridsPy ecosystem."
    )
    st.markdown(
        "It supports off-grid and weak-grid mini-grid design, combining renewable generation, batteries, generators, "
        "optional grid interaction, stochastic scenarios, and both typical-year and multi-year capacity-expansion formulations. "
        "The broader ecosystem connects resource assessment, demand modelling, distribution design, planning, and detailed operational analysis into one coherent workflow."
    )

    _render_featured_planning_card()
    st.subheader("Ecosystem tools")
    row1 = st.columns(2, gap="large")
    with row1[0]:
        _tool_card(
            title="RAMP Demand Model",
            description="Bottom-up stochastic demand assessment for appliances, households, and community load evolution.",
            image_name="ramp_tool_card.png",
            badge="Upstream input layer",
            repo_url=RAMP_URL,
        )
    with row1[1]:
        _tool_card(
            title="PVGIS Resource Assessment",
            description="Solar and wind resource estimation used to build planning-ready renewable input profiles.",
            image_name="pvgis_tool_card.png",
            badge="Upstream input layer",
            repo_url=PVGIS_URL,
        )

    row2 = st.columns(2, gap="large")
    with row2[0]:
        _tool_card(
            title="LV Distribution Topology Tool",
            description="Distribution network layout, pole placement, and topology design for the physical electrification layer.",
            image_name="distribution_tool_card.jpeg",
            badge="Network design layer",
            repo_url=LV_TOPOLOGY_URL,
        )
    with row2[1]:
        _tool_card(
            title="Dispatch Simulation Module",
            description="Detailed operational analysis starting from a predefined system design, useful for operational realism and control studies.",
            image_name="simulation_tool_card.png",
            badge="Operational analysis layer",
        )


def _render_resources() -> None:
    st.subheader("Resources and Navigation")
    st.write(
        "Use this application as the planning workspace inside the broader ecosystem. "
        "The links below help you start a new project and locate the main reference material already available in this repository."
    )
    st.write("")

    c1, c2 = st.columns([1, 3.0], gap="large")
    with c1:
        st.markdown("**Start here in this app**")
        st.page_link("pages/0_Project_Setup.py", label="1. Project Setup")
        st.page_link("pages/1_Data_Audit_and_Visualization.py", label="2. Data Audit and Visualization")
        st.page_link("pages/3_Optimization.py", label="3. Optimization")
        st.page_link("pages/4_Results.py", label="4. Results")

    with c2:
        st.markdown("**Repository references**")
        st.markdown(
            """
            - `README.md`: overall project scope and workflow
            - `docs/DATA_CONTRACT.md`: canonical dataset contract
            - `MicroGridsPy___Mathematical_Formulation_2_0.pdf`: formulation reference
            - `projects/`: example projects and input templates
            """
        )

    st.write("")
    st.markdown("**At a glance**")
    info_cols = st.columns(3, gap="medium")
    with info_cols[0]:
        st.info("Planning modes: Typical-year for compact investment studies, multi-year for dynamic expansion and long-horizon planning.")
    with info_cols[1]:
        st.info("Backend: Python + Streamlit frontend, Linopy optimization backend, CSV/YAML/JSON project workflow.")
    with info_cols[2]:
        st.info("Use together: Resource, demand, planning, network, and dispatch modules can be combined at increasing levels of detail.")

    st.write("")
    st.markdown(
        """
        **Useful links**:

        - User Guide (PDF): https://github.com/AleOnori98/microgridspy-planning/blob/main/MicroGridsPy___Mathematical_Formulation_2_0.pdf 
        - Online Documentation (work in progress): https://microgridspy-documentation.readthedocs.io/en/latest/index.html 
        """
    )



def _render_footer() -> None:
    st.subheader("Contacts")
    st.markdown("**Active Developer**")
    st.markdown(
        """
        **Alessandro Onori**  
        *Core Linopy optimization model, modeling advancements, and Streamlit UI development*
        """
    )

    st.markdown("**Technical Advisors**")
    st.markdown(
        """
        - Riccardo Mereu, Politecnico di Milano
        - Emanuela Colombo, Politecnico di Milano
        """
    )

    st.subheader("License")
    st.markdown("Open-source research codebase. Refer to the repository materials for the current licensing terms.")


def render_home_page() -> None:
    _inject_css()
    _render_ecosystem()
    _render_resources()
    st.divider()
    _render_footer()


render_home_page()
