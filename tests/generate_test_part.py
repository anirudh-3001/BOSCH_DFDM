"""
Generate sample STEP files for testing the DfM analysis pipeline.

Creates a realistic test bracket with:
- Planar faces (flat walls, top/bottom)
- Cylindrical faces (mounting holes)
- Fillet faces (rounded edges - torus/bspline surfaces)

Uses CadQuery for easy parametric part creation, then exports to STEP.

Usage:
    python tests/generate_test_part.py

Output:
    samples/test_bracket.stp
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_bracket() -> None:
    """Generate a test bracket STEP file using CadQuery.

    The bracket is a rectangular base with:
    1. A raised boss on top
    2. Two mounting holes (through-holes)
    3. Filleted edges

    This creates a mix of surface types:
    - Plane (flat faces)
    - Cylinder (holes)
    - Torus/BSpline (fillets)
    """
    import cadquery as cq

    print("[BUILD] Generating test bracket with CadQuery...")

    # ---------------------------------------------------------------
    # Build the bracket parametrically
    # ---------------------------------------------------------------
    bracket = (
        cq.Workplane("XY")
        # Step 1: Base plate - 80 x 40 x 10 mm
        .box(80, 40, 10)
        # Step 2: Add a raised boss on top - 30 x 20 x 15 mm
        .faces(">Z")
        .workplane()
        .rect(30, 20)
        .extrude(15)
        # Step 3: Add two mounting holes (dia 8mm, through the base)
        .faces("<Z")
        .workplane(invert=True)
        .pushPoints([(-25, 0), (25, 0)])
        .hole(8, 10)
        # Step 4: Fillet edges (radius 1.5mm)
        .edges()
        .fillet(1.5)
    )

    # ---------------------------------------------------------------
    # Export to STEP
    # ---------------------------------------------------------------
    output_dir = project_root / "samples"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_bracket.stp"

    cq.exporters.export(bracket, str(output_path), cq.exporters.ExportTypes.STEP)

    file_size = output_path.stat().st_size
    print(f"   [OK] STEP file saved: {output_path}")
    print(f"   [INFO] File size: {file_size:,} bytes")

    # ---------------------------------------------------------------
    # Validate - try loading it back with our parser
    # ---------------------------------------------------------------
    from src.cad.step_parser import STEPParser

    parser = STEPParser()
    shape = parser.load(output_path)
    stats = parser.get_topology_counts()

    print(f"\n   [STATS] Validation:")
    print(f"      Faces:  {stats['faces']}")
    print(f"      Edges:  {stats['edges']}")
    print(f"      Solids: {stats['solids']}")
    print(f"\n   [DONE] Test bracket generated successfully!")


def generate_simple_box() -> None:
    """Generate a minimal box STEP file for basic testing."""
    import cadquery as cq

    print("[BUILD] Generating simple box...")

    box = cq.Workplane("XY").box(50, 30, 20)

    output_dir = project_root / "samples"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "simple_box.stp"

    cq.exporters.export(box, str(output_path), cq.exporters.ExportTypes.STEP)

    print(f"   [OK] STEP file saved: {output_path}")
    print(f"   [DONE] Simple box generated!")


if __name__ == "__main__":
    generate_bracket()
    print()
    generate_simple_box()
