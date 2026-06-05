"""
Shape tessellator — converts OpenCascade B-Rep geometry to triangle meshes.

This is the BRIDGE between the CAD world (B-Rep = exact mathematical surfaces)
and the visualization/analysis world (triangulated meshes).

Theory:
    A STEP file stores geometry as B-Rep (Boundary Representation):
    - Each face is an exact mathematical surface (plane, cylinder, B-spline, etc.)
    - These surfaces cannot be rendered directly — they must be approximated
      by triangles (tessellation / meshing).

    OpenCascade's BRepMesh_IncrementalMesh does this tessellation.
    We control quality via two parameters:
    - linear_deflection:  max distance between mesh and true surface (mm)
    - angular_deflection: max angle between adjacent triangle normals (radians)

    Lower values = finer mesh = more triangles = better quality but slower.

Adapted for OCP (CadQuery's OpenCascade bindings):
    - Import paths: OCP.X instead of OCC.Core.X
    - Static methods use _s suffix: e.g., BRepBndLib.Add_s()
    - TopoDS cast functions: TopoDS.Face_s() instead of topods.Face()

Usage:
    >>> from src.cad.step_parser import STEPParser
    >>> from src.cad.tessellator import ShapeTessellator
    >>>
    >>> parser = STEPParser()
    >>> shape = parser.load("part.stp")
    >>>
    >>> tessellator = ShapeTessellator()
    >>> model = tessellator.tessellate(shape)
    >>> print(model)
    Model(faces=42, edges=63, triangles=1248, area=12345.67)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.GCPnts import GCPnts_UniformDeflection
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp as topexp
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.gp import gp_Pnt, gp_Vec

from src.models.geometry import (
    EdgeTessellation,
    FaceTessellation,
    ModelTessellation,
    SurfaceType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenCascade surface type → our SurfaceType enum
# ---------------------------------------------------------------------------
_OCC_SURFACE_TYPE_MAP = {
    GeomAbs_Plane: SurfaceType.PLANE,
    GeomAbs_Cylinder: SurfaceType.CYLINDER,
    GeomAbs_Cone: SurfaceType.CONE,
    GeomAbs_Sphere: SurfaceType.SPHERE,
    GeomAbs_Torus: SurfaceType.TORUS,
    GeomAbs_BezierSurface: SurfaceType.BEZIER,
    GeomAbs_BSplineSurface: SurfaceType.BSPLINE,
    GeomAbs_SurfaceOfRevolution: SurfaceType.REVOLUTION,
    GeomAbs_SurfaceOfExtrusion: SurfaceType.EXTRUSION,
    GeomAbs_OffsetSurface: SurfaceType.OFFSET,
    GeomAbs_OtherSurface: SurfaceType.OTHER,
}


class ShapeTessellator:
    """Converts an OpenCascade TopoDS_Shape into a ModelTessellation.

    This class handles:
    1. Meshing the B-Rep shape into triangles (BRepMesh_IncrementalMesh)
    2. Extracting per-face triangle data (vertices, indices, normals)
    3. Extracting edge polylines for wireframe display
    4. Computing geometric properties (area, center, surface type)

    Args:
        linear_deflection:  Max chord deviation from true surface (model units).
                            Smaller = finer mesh. Default 0.1 is good for mm-scale parts.
        angular_deflection: Max angular deviation between adjacent normals (radians).
                            Default 0.5 rad ≈ 28.6°.
    """

    def __init__(
        self,
        linear_deflection: float = 0.1,
        angular_deflection: float = 0.5,
    ) -> None:
        self.linear_deflection = linear_deflection
        self.angular_deflection = angular_deflection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tessellate(self, shape: TopoDS_Shape) -> ModelTessellation:
        """Tessellate a shape and return the complete model data.

        Args:
            shape: The OpenCascade shape to tessellate.

        Returns:
            ModelTessellation containing all faces, edges, and metadata.

        Raises:
            RuntimeError: If tessellation fails.
        """
        logger.info(
            "Tessellating shape (linear_deflection=%.4f, angular_deflection=%.4f)",
            self.linear_deflection,
            self.angular_deflection,
        )

        # --- Step 1: Mesh the entire shape ---
        mesh = BRepMesh_IncrementalMesh(
            shape,
            self.linear_deflection,
            False,  # isRelative
            self.angular_deflection,
            True,   # isInParallel
        )
        mesh.Perform()

        if not mesh.IsDone():
            raise RuntimeError(
                "BRepMesh_IncrementalMesh failed. The shape may be invalid."
            )

        # --- Step 2: Extract face tessellations ---
        face_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes_s(shape, TopAbs_FACE, face_map)
        num_faces = face_map.Extent()
        logger.info("Extracting %d face(s)", num_faces)

        faces: list[FaceTessellation] = []
        for i in range(1, num_faces + 1):
            face = TopoDS.Face_s(face_map.FindKey(i))
            face_data = self._extract_face(face, face_id=i - 1)
            if face_data is not None:
                faces.append(face_data)
            else:
                logger.warning("Face %d could not be tessellated (skipped)", i - 1)

        # --- Step 3: Extract edge tessellations ---
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes_s(shape, TopAbs_EDGE, edge_map)
        num_edges = edge_map.Extent()
        logger.info("Extracting %d edge(s)", num_edges)

        edges: list[EdgeTessellation] = []
        for i in range(1, num_edges + 1):
            edge = TopoDS.Edge_s(edge_map.FindKey(i))
            edge_data = self._extract_edge(edge, edge_id=i - 1)
            if edge_data is not None:
                edges.append(edge_data)

        # --- Step 4: Compute metadata ---
        bbox = self._get_bounding_box(shape)
        total_verts = sum(len(f.vertices) for f in faces)
        total_tris = sum(len(f.triangles) for f in faces)

        model = ModelTessellation(
            faces=faces,
            edges=edges,
            total_vertices=total_verts,
            total_triangles=total_tris,
            bounding_box=bbox,
        )

        logger.info("Tessellation complete: %s", model)
        return model

    # ------------------------------------------------------------------
    # Face extraction
    # ------------------------------------------------------------------

    def _extract_face(self, face, face_id: int) -> Optional[FaceTessellation]:
        """Extract tessellation data from a single B-Rep face.

        Args:
            face:    The OpenCascade face (TopoDS_Face).
            face_id: Integer ID to assign.

        Returns:
            FaceTessellation or None if the face has no triangulation.
        """
        try:
            # --- Get the triangulation ---
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)

            if triangulation is None:
                return None

            trsf = location.Transformation()
            is_identity = location.IsIdentity()

            # --- Extract vertices ---
            nb_nodes = triangulation.NbNodes()
            vertices = np.empty((nb_nodes, 3), dtype=np.float64)

            for i in range(1, nb_nodes + 1):
                pnt = triangulation.Node(i)
                if not is_identity:
                    pnt = pnt.Transformed(trsf)
                vertices[i - 1] = [pnt.X(), pnt.Y(), pnt.Z()]

            # --- Extract triangles ---
            nb_tris = triangulation.NbTriangles()
            triangles = np.empty((nb_tris, 3), dtype=np.int32)

            for i in range(1, nb_tris + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                # Convert from 1-indexed to 0-indexed
                triangles[i - 1] = [n1 - 1, n2 - 1, n3 - 1]

            # --- Flip triangle winding for reversed faces ---
            # OpenCascade marks faces as REVERSED when the surface normal
            # is opposite to the face's outward direction. We flip the
            # triangle winding so normals computed from triangles are correct.
            if face.Orientation() == TopAbs_REVERSED:
                triangles = triangles[:, ::-1]

            # --- Compute surface type ---
            surface_type = self._get_surface_type(face)

            # --- Compute area and center of mass ---
            area, center = self._compute_face_properties(face)

            # --- Compute outward-pointing face normal ---
            normal = self._compute_face_normal(face, vertices, triangles)

            return FaceTessellation(
                face_id=face_id,
                vertices=vertices,
                triangles=triangles,
                normal=normal,
                surface_type=surface_type,
                area=area,
                center=center,
            )

        except Exception as exc:
            logger.error("Error extracting face %d: %s", face_id, exc)
            return None

    # ------------------------------------------------------------------
    # Edge extraction
    # ------------------------------------------------------------------

    def _extract_edge(self, edge, edge_id: int) -> Optional[EdgeTessellation]:
        """Extract a polyline approximation of a B-Rep edge.

        Uses GCPnts_UniformDeflection to discretize the edge curve into
        a series of points suitable for wireframe rendering.
        """
        try:
            curve_adaptor = BRepAdaptor_Curve(edge)

            # Discretize the curve
            discretizer = GCPnts_UniformDeflection()
            discretizer.Initialize(
                curve_adaptor,
                self.linear_deflection,
            )

            if not discretizer.IsDone() or discretizer.NbPoints() < 2:
                return None

            nb_points = discretizer.NbPoints()
            points = np.empty((nb_points, 3), dtype=np.float64)

            for i in range(1, nb_points + 1):
                pnt = discretizer.Value(i)
                points[i - 1] = [pnt.X(), pnt.Y(), pnt.Z()]

            return EdgeTessellation(edge_id=edge_id, points=points)

        except Exception as exc:
            logger.debug("Edge %d extraction failed (non-critical): %s", edge_id, exc)
            return None

    # ------------------------------------------------------------------
    # Geometric property helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_surface_type(face) -> SurfaceType:
        """Determine the geometric type of a face's underlying surface."""
        try:
            surf = BRepAdaptor_Surface(face, True)
            geom_type = surf.GetType()
            return _OCC_SURFACE_TYPE_MAP.get(geom_type, SurfaceType.OTHER)
        except Exception:
            return SurfaceType.OTHER

    @staticmethod
    def _compute_face_properties(face) -> tuple[float, np.ndarray]:
        """Compute the surface area and center of mass of a face.

        Uses OpenCascade's GProp system for exact (not approximate) results.

        Returns:
            Tuple of (area, center_array).
        """
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)

        area = props.Mass()
        center_pnt = props.CentreOfMass()
        center = np.array(
            [center_pnt.X(), center_pnt.Y(), center_pnt.Z()], dtype=np.float64
        )

        return area, center

    @staticmethod
    def _compute_face_normal(
        face,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> np.ndarray:
        """Compute the outward-pointing normal of a face.

        Strategy:
        1. Try analytical normal from BRepAdaptor_Surface at face center.
        2. Fall back to area-weighted average of triangle normals.

        The analytical approach is more accurate for curved surfaces.
        The triangle-based fallback handles degenerate cases.
        """
        # --- Approach 1: Analytical normal from surface ---
        try:
            surf = BRepAdaptor_Surface(face, True)
            u_min = surf.FirstUParameter()
            u_max = surf.LastUParameter()
            v_min = surf.FirstVParameter()
            v_max = surf.LastVParameter()

            # Clamp infinite parameters (unbounded planes, etc.)
            u_min = max(u_min, -1e6)
            u_max = min(u_max, 1e6)
            v_min = max(v_min, -1e6)
            v_max = min(v_max, 1e6)

            u_mid = (u_min + u_max) / 2.0
            v_mid = (v_min + v_max) / 2.0

            pnt = gp_Pnt()
            d1u = gp_Vec()
            d1v = gp_Vec()
            surf.D1(u_mid, v_mid, pnt, d1u, d1v)

            normal_vec = d1u.Crossed(d1v)
            magnitude = normal_vec.Magnitude()

            if magnitude > 1e-10:
                normal_vec.Normalize()

                # BRepAdaptor_Surface does NOT account for face orientation
                # in its D1 output. Reversed faces need the normal flipped.
                if face.Orientation() == TopAbs_REVERSED:
                    normal_vec.Reverse()

                return np.array(
                    [normal_vec.X(), normal_vec.Y(), normal_vec.Z()],
                    dtype=np.float64,
                )
        except Exception:
            pass

        # --- Approach 2: Area-weighted triangle normal average ---
        if len(triangles) > 0 and len(vertices) > 0:
            weighted_normal = np.zeros(3, dtype=np.float64)
            for tri in triangles:
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                weighted_normal += cross

            norm = np.linalg.norm(weighted_normal)
            if norm > 1e-10:
                return weighted_normal / norm

        # --- Fallback: default normal ---
        logger.warning("Could not compute normal, defaulting to +Z")
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    @staticmethod
    def _get_bounding_box(
        shape: TopoDS_Shape,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute the axis-aligned bounding box."""
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        return bbox.Get()
