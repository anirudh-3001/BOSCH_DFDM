"""
3D model viewer using Plotly.

Creates interactive 3D visualizations of tessellated CAD models.
Plotly is used instead of PyVista because:
1. Native Streamlit integration (no extra plugins)
2. WebGL-based — runs in any browser
3. Built-in zoom, pan, rotate, hover tooltips
4. Export to PNG/SVG

Color Modes:
    - "surface_type": Color faces by geometric type (Plane=green, Cylinder=blue, etc.)
    - "face_id":      Unique color per face for identification
    - "uniform":      Single color for the entire model

Usage:
    >>> viewer = ModelViewer()
    >>> fig = viewer.create_figure(model, color_mode="surface_type", show_edges=True)
    >>> fig.show()  # or st.plotly_chart(fig)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from src.models.geometry import (
    ModelTessellation,
    SurfaceType,
    SURFACE_TYPE_COLORS,
)


# ---------------------------------------------------------------------------
# Generate distinct colors for face_id mode
# ---------------------------------------------------------------------------

def _generate_face_colors(num_faces: int) -> list[str]:
    """Generate a list of visually distinct colors using HSL rotation."""
    colors = []
    for i in range(num_faces):
        hue = (i * 137.508) % 360  # Golden angle for maximum separation
        colors.append(f"hsl({hue:.0f}, 70%, 55%)")
    return colors


class ModelViewer:
    """Creates Plotly 3D figures from tessellated CAD models.

    This viewer supports multiple color modes and optional wireframe overlay.
    It is designed to integrate seamlessly with Streamlit.
    """

    # Dark theme scene configuration
    SCENE_CONFIG = dict(
        bgcolor="#0E1117",
        xaxis=dict(
            backgroundcolor="#0E1117",
            gridcolor="#2a2a3e",
            color="#888",
            showspikes=False,
            title="X",
        ),
        yaxis=dict(
            backgroundcolor="#0E1117",
            gridcolor="#2a2a3e",
            color="#888",
            showspikes=False,
            title="Y",
        ),
        zaxis=dict(
            backgroundcolor="#0E1117",
            gridcolor="#2a2a3e",
            color="#888",
            showspikes=False,
            title="Z",
        ),
        aspectmode="data",
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=1.5, y=1.5, z=1.0),
        ),
    )

    LAYOUT_CONFIG = dict(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        showlegend=False,
        font=dict(color="#E0E0E0", family="Inter, sans-serif"),
    )

    def create_figure(
        self,
        model: ModelTessellation,
        color_mode: str = "surface_type",
        show_edges: bool = True,
        show_normals: bool = False,
        opacity: float = 1.0,
    ) -> go.Figure:
        """Create an interactive 3D figure of the model.

        Args:
            model:        The tessellated model to visualize.
            color_mode:   One of "surface_type", "face_id", "uniform".
            show_edges:   Whether to overlay wireframe edges.
            show_normals: Whether to show face normal arrows.
            opacity:      Surface opacity (0.0–1.0).

        Returns:
            A Plotly Figure object ready for display.
        """
        fig = go.Figure()

        # --- Add mesh surface ---
        mesh_trace = self._build_mesh_trace(model, color_mode, opacity)
        fig.add_trace(mesh_trace)

        # --- Add wireframe edges ---
        if show_edges and model.edges:
            edge_trace = self._build_edge_trace(model)
            fig.add_trace(edge_trace)

        # --- Add face normal arrows ---
        if show_normals:
            normal_trace = self._build_normal_trace(model)
            fig.add_trace(normal_trace)

        # --- Apply layout ---
        fig.update_layout(
            scene=self.SCENE_CONFIG,
            **self.LAYOUT_CONFIG,
        )

        return fig

    # ------------------------------------------------------------------
    # Mesh trace
    # ------------------------------------------------------------------

    def _build_mesh_trace(
        self,
        model: ModelTessellation,
        color_mode: str,
        opacity: float,
    ) -> go.Mesh3d:
        """Build the Mesh3d trace with per-triangle coloring."""
        # Combine all face meshes into one big mesh
        all_x: list[float] = []
        all_y: list[float] = []
        all_z: list[float] = []
        all_i: list[int] = []
        all_j: list[int] = []
        all_k: list[int] = []
        face_colors: list[str] = []

        # Pre-compute colors for face_id mode
        id_colors = _generate_face_colors(model.total_faces)

        vertex_offset = 0

        for idx, face in enumerate(model.faces):
            verts = face.vertices
            tris = face.triangles

            all_x.extend(verts[:, 0].tolist())
            all_y.extend(verts[:, 1].tolist())
            all_z.extend(verts[:, 2].tolist())

            all_i.extend((tris[:, 0] + vertex_offset).tolist())
            all_j.extend((tris[:, 1] + vertex_offset).tolist())
            all_k.extend((tris[:, 2] + vertex_offset).tolist())

            # Determine color for this face's triangles
            if color_mode == "surface_type":
                color = SURFACE_TYPE_COLORS.get(face.surface_type, "#9E9E9E")
            elif color_mode == "face_id":
                color = id_colors[idx % len(id_colors)]
            else:  # uniform
                color = "#00D4FF"

            face_colors.extend([color] * len(tris))
            vertex_offset += len(verts)

        # Build hover text per triangle
        hover_texts = []
        for face in model.faces:
            text = (
                f"Face {face.face_id}<br>"
                f"Type: {face.surface_type.value}<br>"
                f"Area: {face.area:.4f}<br>"
                f"Normal: [{face.normal[0]:.3f}, {face.normal[1]:.3f}, {face.normal[2]:.3f}]"
            )
            hover_texts.extend([text] * len(face.triangles))

        return go.Mesh3d(
            x=all_x,
            y=all_y,
            z=all_z,
            i=all_i,
            j=all_j,
            k=all_k,
            facecolor=face_colors,
            opacity=opacity,
            flatshading=True,
            hovertext=hover_texts,
            hoverinfo="text",
            name="Model Surface",
            lighting=dict(
                ambient=0.4,
                diffuse=0.6,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2,
            ),
            lightposition=dict(x=100, y=200, z=300),
        )

    # ------------------------------------------------------------------
    # Edge (wireframe) trace
    # ------------------------------------------------------------------

    @staticmethod
    def _build_edge_trace(model: ModelTessellation) -> go.Scatter3d:
        """Build wireframe edges as a single Scatter3d trace.

        Uses None values to create gaps between separate edges.
        """
        x: list[Optional[float]] = []
        y: list[Optional[float]] = []
        z: list[Optional[float]] = []

        for edge in model.edges:
            pts = edge.points
            x.extend(pts[:, 0].tolist() + [None])
            y.extend(pts[:, 1].tolist() + [None])
            z.extend(pts[:, 2].tolist() + [None])

        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="#FFFFFF", width=1.5),
            name="Edges",
            hoverinfo="skip",
        )

    # ------------------------------------------------------------------
    # Normal arrows trace
    # ------------------------------------------------------------------

    @staticmethod
    def _build_normal_trace(model: ModelTessellation) -> go.Cone:
        """Build face normal arrows using Plotly's Cone trace.

        Each arrow starts at the face center and points in the normal direction.
        """
        # Compute arrow scale based on bounding box
        dims = model.dimensions
        scale = max(dims) * 0.05  # Arrow length = 5% of model size

        x = [f.center[0] for f in model.faces]
        y = [f.center[1] for f in model.faces]
        z = [f.center[2] for f in model.faces]
        u = [f.normal[0] * scale for f in model.faces]
        v = [f.normal[1] * scale for f in model.faces]
        w = [f.normal[2] * scale for f in model.faces]

        return go.Cone(
            x=x, y=y, z=z,
            u=u, v=v, w=w,
            sizemode="absolute",
            sizeref=scale * 0.3,
            colorscale=[[0, "#FF6B6B"], [1, "#FF6B6B"]],
            showscale=False,
            name="Face Normals",
            hoverinfo="skip",
            anchor="tail",
        )

    # ------------------------------------------------------------------
    # Utility: Surface type legend figure
    # ------------------------------------------------------------------

    @staticmethod
    def create_surface_type_chart(model: ModelTessellation) -> go.Figure:
        """Create a pie chart showing surface type distribution."""
        dist = model.surface_type_distribution()

        labels = [st.value for st in dist.keys()]
        values = list(dist.values())
        colors = [SURFACE_TYPE_COLORS.get(st, "#9E9E9E") for st in dist.keys()]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors),
                    textinfo="label+value",
                    textfont=dict(color="#E0E0E0", size=12),
                    hole=0.4,
                    hovertemplate="%{label}: %{value} faces<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#E0E0E0", family="Inter, sans-serif"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            showlegend=True,
            legend=dict(
                font=dict(color="#E0E0E0", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        return fig
