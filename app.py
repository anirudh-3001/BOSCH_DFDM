"""
Bosch DfM Agent — STEP Viewer (Phase 1)

A professional Streamlit dashboard for uploading, parsing, and
interactively visualizing STEP (.stp) CAD files.

Run with:
    streamlit run app.py

Features:
    - Drag-and-drop STEP file upload
    - Interactive 3D model viewer (Plotly)
    - Color by surface type or face ID
    - Wireframe overlay toggle
    - Face normal arrows toggle
    - Model statistics (faces, edges, area, bounding box)
    - Surface type distribution chart
    - Detailed face data table
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.cad.step_parser import STEPParser
from src.cad.tessellator import ShapeTessellator
from src.models.geometry import ModelTessellation, SURFACE_TYPE_COLORS
from src.visualization.viewer import ModelViewer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bosch DfM Agent — STEP Viewer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 50%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 212, 255, 0.15);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        color: #00D4FF;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #8892b0;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.3s ease;
    }

    .stat-card:hover {
        border-color: rgba(0, 212, 255, 0.4);
    }

    .stat-label {
        color: #8892b0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
    }

    .stat-value {
        color: #00D4FF;
        font-size: 1.4rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        color: #E0E0E0;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid #00D4FF;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .status-success {
        background: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }

    .status-waiting {
        background: rgba(255, 152, 0, 0.15);
        color: #FF9800;
        border: 1px solid rgba(255, 152, 0, 0.3);
    }

    /* Hide Streamlit default header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def render_stat_card(label: str, value: str) -> None:
    """Render a styled statistic card."""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the application header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🏭 Bosch DfM Agent</h1>
            <p>AI-driven Design for Manufacturability Analysis Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Caching: parse & tessellate once per file
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_tessellate(file_bytes: bytes, file_name: str) -> ModelTessellation:
    """Load a STEP file from bytes and return the tessellated model.

    This function is cached by Streamlit — subsequent calls with the
    same file bytes skip re-parsing (instant reload on settings change).
    """
    # Write bytes to a temp file (pythonocc needs a file path)
    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    parser = STEPParser()
    shape = parser.load(tmp_path)

    tessellator = ShapeTessellator(
        linear_deflection=0.1,
        angular_deflection=0.5,
    )
    model = tessellator.tessellate(shape)

    return model


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry point."""
    render_header()

    # ===================================================================
    # SIDEBAR
    # ===================================================================
    with st.sidebar:
        st.markdown("### 📁 Upload STEP File")
        uploaded_file = st.file_uploader(
            "Drag and drop a .stp or .step file",
            type=["stp", "step"],
            help="Upload a STEP file to analyze. Max 200 MB.",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Display options (shown only when model is loaded)
        if "model" in st.session_state:
            st.markdown("### 🎨 Display Options")

            color_mode = st.selectbox(
                "Color by",
                options=["surface_type", "face_id", "uniform"],
                format_func=lambda x: {
                    "surface_type": "🎨 Surface Type",
                    "face_id": "🔢 Face ID",
                    "uniform": "⬜ Uniform",
                }[x],
                help="Choose how to color the model faces.",
            )

            show_edges = st.checkbox("🔲 Show wireframe edges", value=True)
            show_normals = st.checkbox("➡️ Show face normals", value=False)

            opacity = st.slider(
                "Surface opacity",
                min_value=0.3,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )

            st.markdown("---")

    # ===================================================================
    # MAIN CONTENT
    # ===================================================================

    if uploaded_file is None:
        # --- No file uploaded: show instructions ---
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📐</div>
                <h2 style="color: #E0E0E0; font-weight: 600;">Upload a STEP File to Begin</h2>
                <p style="color: #8892b0; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                    Drag and drop a <code>.stp</code> or <code>.step</code> file into the
                    sidebar to parse, visualize, and analyze its geometry.
                </p>
                <div style="margin-top: 2rem; color: #555;">
                    <p style="font-size: 0.85rem;">
                        💡 Don't have a STEP file? Run
                        <code>python tests/generate_test_part.py</code>
                        to create a sample.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # --- File uploaded: parse and display ---
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

    # Load and tessellate (cached)
    with st.spinner("🔧 Parsing STEP geometry..."):
        try:
            model = load_and_tessellate(file_bytes, file_name)
            st.session_state["model"] = model
        except Exception as exc:
            st.error(f"❌ Failed to load STEP file: {exc}")
            logger.exception("STEP loading failed")
            return

    # --- Success badge ---
    st.markdown(
        f"""
        <span class="status-badge status-success">✓ Loaded</span>
        <span style="color: #8892b0; margin-left: 0.5rem;">{file_name}</span>
        """,
        unsafe_allow_html=True,
    )

    # ===================================================================
    # SIDEBAR: Model Statistics
    # ===================================================================
    with st.sidebar:
        st.markdown("### 📊 Model Statistics")

        render_stat_card("Total Faces", str(model.total_faces))
        render_stat_card("Total Edges", str(model.total_edges))
        render_stat_card("Total Triangles", f"{model.total_triangles:,}")
        render_stat_card("Surface Area", f"{model.total_area:.2f}")

        dims = model.dimensions
        render_stat_card(
            "Bounding Box",
            f"{dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f}",
        )

        bbox = model.bounding_box
        render_stat_card(
            "Origin Range",
            f"({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}) → "
            f"({bbox[3]:.1f}, {bbox[4]:.1f}, {bbox[5]:.1f})",
        )

    # ===================================================================
    # 3D VIEWER
    # ===================================================================
    st.markdown('<div class="section-header">🖥️ Interactive 3D Viewer</div>', unsafe_allow_html=True)

    viewer = ModelViewer()
    fig = viewer.create_figure(
        model,
        color_mode=color_mode,
        show_edges=show_edges,
        show_normals=show_normals,
        opacity=opacity,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # ===================================================================
    # SURFACE TYPE DISTRIBUTION
    # ===================================================================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            '<div class="section-header">📊 Surface Type Distribution</div>',
            unsafe_allow_html=True,
        )
        pie_fig = viewer.create_surface_type_chart(model)
        st.plotly_chart(pie_fig, use_container_width=True)

    # ===================================================================
    # FACE DATA TABLE
    # ===================================================================
    with col2:
        st.markdown(
            '<div class="section-header">📋 Face Details</div>',
            unsafe_allow_html=True,
        )

        # Build a DataFrame of face properties
        face_data = []
        for face in model.faces:
            face_data.append(
                {
                    "Face ID": face.face_id,
                    "Surface Type": face.surface_type.value,
                    "Area": round(face.area, 4),
                    "Normal X": round(face.normal[0], 4),
                    "Normal Y": round(face.normal[1], 4),
                    "Normal Z": round(face.normal[2], 4),
                    "Center X": round(face.center[0], 2),
                    "Center Y": round(face.center[1], 2),
                    "Center Z": round(face.center[2], 2),
                    "Triangles": len(face.triangles),
                }
            )

        df = pd.DataFrame(face_data)
        st.dataframe(
            df,
            use_container_width=True,
            height=350,
            hide_index=True,
        )

    # ===================================================================
    # SURFACE TYPE SUMMARY TABLE
    # ===================================================================
    st.markdown(
        '<div class="section-header">📈 Surface Type Summary</div>',
        unsafe_allow_html=True,
    )

    dist = model.surface_type_distribution()
    summary_data = []
    for stype, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        # Compute total area for this surface type
        type_area = sum(f.area for f in model.faces if f.surface_type == stype)
        summary_data.append(
            {
                "Surface Type": stype.value,
                "Face Count": count,
                "Percentage": f"{count / model.total_faces * 100:.1f}%",
                "Total Area": round(type_area, 2),
                "Area %": f"{type_area / model.total_area * 100:.1f}%",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
