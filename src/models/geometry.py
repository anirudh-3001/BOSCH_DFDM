"""
Geometry data models for the DfM analysis pipeline.

These dataclasses represent the tessellated (triangulated) geometry extracted
from STEP files. They serve as the common data exchange format between all
analysis engines (draft, undercut, parting line, etc.).

Design Decision:
    We use NumPy arrays (not Python lists) for vertex/triangle data because:
    1. Memory efficient for large models (1000+ faces)
    2. Vectorized math operations for draft/normal calculations
    3. Direct compatibility with Plotly and PyVista visualization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SurfaceType(Enum):
    """Classification of B-Rep surface types from OpenCascade.

    Each face in a STEP model has an underlying geometric surface.
    Knowing the surface type is critical for DfM analysis:
    - PLANE faces are easy to mold (flat walls, tops, bottoms)
    - CYLINDER/CONE faces often form holes, bosses, ribs
    - BSPLINE faces are freeform surfaces (complex mold geometry)
    - TORUS faces typically appear at fillets/rounds
    """

    PLANE = "Plane"
    CYLINDER = "Cylinder"
    CONE = "Cone"
    SPHERE = "Sphere"
    TORUS = "Torus"
    BEZIER = "BezierSurface"
    BSPLINE = "BSplineSurface"
    REVOLUTION = "SurfaceOfRevolution"
    EXTRUSION = "SurfaceOfExtrusion"
    OFFSET = "OffsetSurface"
    OTHER = "OtherSurface"


# ---------------------------------------------------------------------------
# Color palettes for visualization
# ---------------------------------------------------------------------------

SURFACE_TYPE_COLORS: dict[SurfaceType, str] = {
    SurfaceType.PLANE: "#4CAF50",       # Green — flat faces
    SurfaceType.CYLINDER: "#2196F3",    # Blue — cylindrical holes/bosses
    SurfaceType.CONE: "#FF9800",        # Orange — tapered features
    SurfaceType.SPHERE: "#9C27B0",      # Purple — spherical blends
    SurfaceType.TORUS: "#F44336",       # Red — fillets/rounds
    SurfaceType.BEZIER: "#CDDC39",      # Lime — bezier freeform
    SurfaceType.BSPLINE: "#00BCD4",     # Cyan — bspline freeform
    SurfaceType.REVOLUTION: "#795548",  # Brown — revolved features
    SurfaceType.EXTRUSION: "#607D8B",   # Blue Grey — extruded features
    SurfaceType.OFFSET: "#E91E63",      # Pink — offset surfaces
    SurfaceType.OTHER: "#9E9E9E",       # Grey — unknown
}


@dataclass
class FaceTessellation:
    """Tessellated representation of a single B-Rep face.

    After OpenCascade meshes the shape, each face is decomposed into
    triangles. This class stores that triangulation along with geometric
    properties needed for DfM analysis.

    Attributes:
        face_id:      Unique integer ID for this face (0-indexed).
        vertices:     (N, 3) array of triangle vertex coordinates [x, y, z].
        triangles:    (M, 3) array of triangle vertex indices (into `vertices`).
        normal:       (3,) unit vector — outward-pointing face normal at center.
        surface_type: Geometric classification of the underlying surface.
        area:         Surface area of the face (in model units²).
        center:       (3,) center of mass of the face.
    """

    face_id: int
    vertices: np.ndarray       # shape (N, 3)
    triangles: np.ndarray      # shape (M, 3), dtype int
    normal: np.ndarray         # shape (3,)
    surface_type: SurfaceType
    area: float
    center: np.ndarray         # shape (3,)

    def __repr__(self) -> str:
        return (
            f"Face(id={self.face_id}, type={self.surface_type.value}, "
            f"area={self.area:.4f}, triangles={len(self.triangles)})"
        )


@dataclass
class EdgeTessellation:
    """Tessellated representation of a single B-Rep edge.

    Edges are discretized into polylines for wireframe display.
    In later phases, edges are also used for parting line detection.

    Attributes:
        edge_id: Unique integer ID for this edge (0-indexed).
        points:  (N, 3) array of polyline point coordinates.
    """

    edge_id: int
    points: np.ndarray  # shape (N, 3)

    def __repr__(self) -> str:
        return f"Edge(id={self.edge_id}, points={len(self.points)})"


@dataclass
class ModelTessellation:
    """Complete tessellated model — the central data structure.

    This is the output of the tessellation pipeline and the input to
    ALL downstream analysis engines (draft, undercut, parting line, etc.).

    Attributes:
        faces:           List of tessellated faces.
        edges:           List of tessellated edges.
        total_vertices:  Total vertex count across all faces.
        total_triangles: Total triangle count across all faces.
        bounding_box:    (xmin, ymin, zmin, xmax, ymax, zmax).
    """

    faces: list[FaceTessellation]
    edges: list[EdgeTessellation]
    total_vertices: int
    total_triangles: int
    bounding_box: tuple[float, float, float, float, float, float]

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def total_faces(self) -> int:
        """Number of faces in the model."""
        return len(self.faces)

    @property
    def total_edges(self) -> int:
        """Number of edges in the model."""
        return len(self.edges)

    @property
    def total_area(self) -> float:
        """Total surface area of all faces."""
        return sum(f.area for f in self.faces)

    @property
    def dimensions(self) -> tuple[float, float, float]:
        """Bounding box dimensions (dx, dy, dz)."""
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounding_box
        return (xmax - xmin, ymax - ymin, zmax - zmin)

    def surface_type_distribution(self) -> dict[SurfaceType, int]:
        """Count of faces by surface type."""
        dist: dict[SurfaceType, int] = {}
        for face in self.faces:
            dist[face.surface_type] = dist.get(face.surface_type, 0) + 1
        return dist

    def get_face_by_id(self, face_id: int) -> Optional[FaceTessellation]:
        """Look up a face by its ID."""
        for face in self.faces:
            if face.face_id == face_id:
                return face
        return None

    def __repr__(self) -> str:
        return (
            f"Model(faces={self.total_faces}, edges={self.total_edges}, "
            f"triangles={self.total_triangles}, area={self.total_area:.2f})"
        )
