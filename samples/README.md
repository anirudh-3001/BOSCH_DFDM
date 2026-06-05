# Sample STEP Files

Place your `.stp` or `.step` files in this directory.

## Generating a Test Part

Run the test part generator to create a sample STEP file:

```bash
python tests/generate_test_part.py
```

This will create `samples/test_bracket.stp` — a simple bracket with:
- Planar faces (flat walls)
- Cylindrical faces (holes)
- Fillet faces (rounded edges)

Perfect for validating the DfM analysis pipeline.
