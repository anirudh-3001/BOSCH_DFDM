"""
Unit tests for the STEP parser and tessellator.

Run with:
    python -m pytest tests/test_step_parser.py -v

These tests require:
    1. The conda environment with pythonocc-core installed
    2. A sample STEP file at samples/test_bracket.stp
       (generate it first with: python tests/generate_test_part.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cad.step_parser import STEPParser
from src.cad.tessellator import ShapeTessellator
from src.models.geometry import (
    FaceTessellation,
    ModelTessellation,
    SurfaceType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FILE = project_root / "samples" / "test_bracket.stp"


@pytest.fixture
def sample_path() -> Path:
    """Path to the sample STEP file."""
    if not SAMPLE_FILE.exists():
        pytest.skip(
            f"Sample file not found: {SAMPLE_FILE}. "
            "Run 'python tests/generate_test_part.py' first."
        )
    return SAMPLE_FILE


@pytest.fixture
def parser() -> STEPParser:
    """Fresh STEPParser instance."""
    return STEPParser()


@pytest.fixture
def loaded_shape(parser: STEPParser, sample_path: Path):
    """Loaded OpenCascade shape from the sample file."""
    return parser.load(sample_path)


@pytest.fixture
def model(loaded_shape) -> ModelTessellation:
    """Tessellated model from the sample file."""
    tessellator = ShapeTessellator()
    return tessellator.tessellate(loaded_shape)


# ---------------------------------------------------------------------------
# STEPParser Tests
# ---------------------------------------------------------------------------

class TestSTEPParser:
    """Tests for STEPParser."""

    def test_load_returns_shape(self, parser: STEPParser, sample_path: Path):
        """Loading a valid STEP file returns a non-None shape."""
        shape = parser.load(sample_path)
        assert shape is not None

    def test_file_not_found(self, parser: STEPParser):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parser.load("nonexistent_file.stp")

    def test_invalid_extension(self, parser: STEPParser, tmp_path: Path):
        """Loading a file with wrong extension raises ValueError."""
        bad_file = tmp_path / "model.stl"
        bad_file.write_text("not a step file")
        with pytest.raises(ValueError, match="Invalid file extension"):
            parser.load(bad_file)

    def test_topology_counts(self, parser: STEPParser, sample_path: Path):
        """Loaded shape has reasonable topology counts."""
        parser.load(sample_path)
        stats = parser.get_topology_counts()

        assert stats["faces"] > 0, "Should have at least one face"
        assert stats["edges"] > 0, "Should have at least one edge"
        assert stats["solids"] >= 1, "Should have at least one solid"

    def test_bounding_box(self, parser: STEPParser, sample_path: Path):
        """Bounding box has positive dimensions."""
        parser.load(sample_path)
        xmin, ymin, zmin, xmax, ymax, zmax = parser.get_bounding_box()

        assert xmax > xmin, "X dimension should be positive"
        assert ymax > ymin, "Y dimension should be positive"
        assert zmax > zmin, "Z dimension should be positive"

    def test_shape_property(self, parser: STEPParser, sample_path: Path):
        """shape property returns the loaded shape."""
        assert parser.shape is None  # Before loading
        parser.load(sample_path)
        assert parser.shape is not None  # After loading


# ---------------------------------------------------------------------------
# ShapeTessellator Tests
# ---------------------------------------------------------------------------

class TestShapeTessellator:
    """Tests for ShapeTessellator."""

    def test_tessellation_produces_faces(self, model: ModelTessellation):
        """Tessellation should produce at least one face."""
        assert model.total_faces > 0

    def test_tessellation_produces_edges(self, model: ModelTessellation):
        """Tessellation should produce edges."""
        assert model.total_edges > 0

    def test_face_has_valid_vertices(self, model: ModelTessellation):
        """Each face should have 3D vertices."""
        for face in model.faces:
            assert face.vertices.ndim == 2
            assert face.vertices.shape[1] == 3
            assert len(face.vertices) > 0

    def test_face_has_valid_triangles(self, model: ModelTessellation):
        """Each face should have valid triangle indices."""
        for face in model.faces:
            assert face.triangles.ndim == 2
            assert face.triangles.shape[1] == 3
            assert len(face.triangles) > 0
            # Indices should be within vertex array bounds
            assert face.triangles.max() < len(face.vertices)
            assert face.triangles.min() >= 0

    def test_face_normal_is_unit_vector(self, model: ModelTessellation):
        """Face normals should be unit vectors (magnitude ≈ 1)."""
        for face in model.faces:
            magnitude = np.linalg.norm(face.normal)
            assert abs(magnitude - 1.0) < 1e-6, (
                f"Face {face.face_id} normal magnitude = {magnitude}"
            )

    def test_face_area_is_positive(self, model: ModelTessellation):
        """Face areas should be positive."""
        for face in model.faces:
            assert face.area > 0, f"Face {face.face_id} has non-positive area"

    def test_surface_types_are_valid(self, model: ModelTessellation):
        """All face surface types should be valid SurfaceType enum values."""
        for face in model.faces:
            assert isinstance(face.surface_type, SurfaceType)

    def test_has_planar_faces(self, model: ModelTessellation):
        """The test bracket should have planar faces."""
        plane_faces = [
            f for f in model.faces if f.surface_type == SurfaceType.PLANE
        ]
        assert len(plane_faces) > 0, "Bracket should have planar faces"

    def test_has_cylindrical_faces(self, model: ModelTessellation):
        """The test bracket should have cylindrical faces (holes)."""
        cyl_faces = [
            f for f in model.faces if f.surface_type == SurfaceType.CYLINDER
        ]
        assert len(cyl_faces) > 0, "Bracket should have cylindrical faces (holes)"

    def test_bounding_box_is_reasonable(self, model: ModelTessellation):
        """Bounding box should match known bracket dimensions (~80x40x25)."""
        dx, dy, dz = model.dimensions
        # The bracket is approximately 80x40x25 mm (base + boss)
        assert 70 < dx < 90, f"X dimension {dx} not in expected range"
        assert 30 < dy < 50, f"Y dimension {dy} not in expected range"
        assert 20 < dz < 35, f"Z dimension {dz} not in expected range"

    def test_total_area_is_reasonable(self, model: ModelTessellation):
        """Total surface area should be positive and reasonable."""
        assert model.total_area > 0
        assert model.total_area < 1e6  # Sanity check

    def test_surface_type_distribution(self, model: ModelTessellation):
        """Surface type distribution should return valid counts."""
        dist = model.surface_type_distribution()
        assert sum(dist.values()) == model.total_faces


# ---------------------------------------------------------------------------
# ModelTessellation Tests
# ---------------------------------------------------------------------------

class TestModelTessellation:
    """Tests for ModelTessellation computed properties."""

    def test_dimensions(self, model: ModelTessellation):
        """Dimensions should match bounding box."""
        bbox = model.bounding_box
        dims = model.dimensions
        assert abs(dims[0] - (bbox[3] - bbox[0])) < 1e-10
        assert abs(dims[1] - (bbox[4] - bbox[1])) < 1e-10
        assert abs(dims[2] - (bbox[5] - bbox[2])) < 1e-10

    def test_get_face_by_id(self, model: ModelTessellation):
        """get_face_by_id returns correct face or None."""
        face = model.get_face_by_id(0)
        assert face is not None
        assert face.face_id == 0

        missing = model.get_face_by_id(99999)
        assert missing is None
