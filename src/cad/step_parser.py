"""
STEP file parser using OCP (CadQuery's OpenCascade bindings).

Responsibilities:
    1. Read a STEP (.stp / .step) file from disk
    2. Transfer the geometry into an OpenCascade TopoDS_Shape
    3. Provide basic shape metadata (bounding box, validity checks)

Design Decision — OCP vs pythonocc-core:
    OCP is the OpenCascade binding bundled with CadQuery. It can be installed
    via pip (no conda required) and provides the same full OpenCascade API.
    Import paths differ: OCC.Core.X → OCP.X

Usage:
    >>> parser = STEPParser()
    >>> shape = parser.load("part.stp")
    >>> print(parser.get_bounding_box())
    (0.0, 0.0, 0.0, 100.0, 50.0, 30.0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopoDS import TopoDS_Shape
from OCP.TopExp import TopExp as topexp
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_ShapeEnum
from OCP.TopTools import TopTools_IndexedMapOfShape

logger = logging.getLogger(__name__)


class STEPParser:
    """Loads and validates STEP files.

    This is the entry point for the entire DfM pipeline. Every analysis
    starts by loading a STEP file through this class.

    Attributes:
        shape:     The loaded OpenCascade shape (None until load() is called).
        file_path: Path to the loaded file (None until load() is called).
    """

    VALID_EXTENSIONS = {".stp", ".step"}

    def __init__(self) -> None:
        self._shape: Optional[TopoDS_Shape] = None
        self._file_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> TopoDS_Shape:
        """Load a STEP file and return the OpenCascade shape.

        Args:
            file_path: Path to the .stp or .step file.

        Returns:
            The TopoDS_Shape representing the entire model.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .stp or .step.
            RuntimeError: If OpenCascade fails to read or transfer the file.
        """
        file_path = Path(file_path)
        self._validate_file(file_path)

        logger.info("Loading STEP file: %s", file_path)

        # --- Read the STEP file ---
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(file_path))

        if status != IFSelect_RetDone:
            raise RuntimeError(
                f"OpenCascade failed to read STEP file (status={status}): {file_path}"
            )

        # --- Transfer all roots into shapes ---
        num_roots = reader.TransferRoots()
        logger.info("Transferred %d root(s) from STEP file", num_roots)

        if num_roots == 0:
            raise RuntimeError(f"No geometry found in STEP file: {file_path}")

        self._shape = reader.OneShape()
        self._file_path = file_path

        # --- Log basic statistics ---
        stats = self.get_topology_counts()
        logger.info(
            "Loaded shape: %d faces, %d edges, %d solids",
            stats["faces"],
            stats["edges"],
            stats["solids"],
        )

        return self._shape

    @property
    def shape(self) -> Optional[TopoDS_Shape]:
        """The loaded OpenCascade shape, or None."""
        return self._shape

    @property
    def file_path(self) -> Optional[Path]:
        """Path to the loaded STEP file, or None."""
        return self._file_path

    def get_bounding_box(self) -> tuple[float, float, float, float, float, float]:
        """Compute the axis-aligned bounding box of the loaded shape.

        Returns:
            Tuple of (xmin, ymin, zmin, xmax, ymax, zmax) in model units.

        Raises:
            RuntimeError: If no shape has been loaded.
        """
        self._ensure_loaded()
        bbox = Bnd_Box()
        BRepBndLib.Add_s(self._shape, bbox)
        return bbox.Get()

    def get_topology_counts(self) -> dict[str, int]:
        """Count topological entities (faces, edges, solids) in the shape.

        Returns:
            Dictionary with keys 'faces', 'edges', 'solids'.
        """
        self._ensure_loaded()

        face_map = TopTools_IndexedMapOfShape()
        edge_map = TopTools_IndexedMapOfShape()
        solid_map = TopTools_IndexedMapOfShape()

        topexp.MapShapes_s(self._shape, TopAbs_FACE, face_map)
        topexp.MapShapes_s(self._shape, TopAbs_EDGE, edge_map)
        topexp.MapShapes_s(self._shape, TopAbs_SOLID, solid_map)

        return {
            "faces": face_map.Extent(),
            "edges": edge_map.Extent(),
            "solids": solid_map.Extent(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_file(self, file_path: Path) -> None:
        """Validate the file exists and has a correct extension."""
        if not file_path.exists():
            raise FileNotFoundError(f"STEP file not found: {file_path}")
        if file_path.suffix.lower() not in self.VALID_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension '{file_path.suffix}'. "
                f"Expected one of: {self.VALID_EXTENSIONS}"
            )

    def _ensure_loaded(self) -> None:
        """Raise if no shape is loaded."""
        if self._shape is None:
            raise RuntimeError(
                "No shape loaded. Call load() first with a valid STEP file."
            )
