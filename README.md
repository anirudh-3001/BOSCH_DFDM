<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCascade-OCP%207.8-0078D4?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Hackathon-green?style=for-the-badge" />
</p>

# 🏭 Bosch DfM Agent

### AI-Driven Design for Manufacturability Analysis for Injection-Molded Plastic Parts

> Automatically analyze STEP (.stp) CAD models to detect **optimal mold opening direction**, **undercuts**, **parting lines**, and **core/cavity classification** — providing actionable manufacturability recommendations backed by computational geometry.

---

## 🎯 Problem Statement

Designing injection-molded plastic parts that are **actually manufacturable** requires deep CAD and mold design expertise. Engineers must manually verify:

- Can the part be pulled out of the mold without getting stuck? (**Undercuts**)
- Where should the mold split? (**Parting Line**)
- Which surfaces belong to the core half vs the cavity half? (**Core/Cavity Classification**)
- Is the selected mold opening direction manufacturable? (**Pull Direction Analysis**)

**This tool automates the analysis pipeline**, reducing manual review time and surfacing manufacturability risks early.

---

## ✨ Features

| Feature | Status | Description |
|---------|--------|-------------|
| 📁 STEP File Parser | ✅ Done | Load and parse `.stp` STEP CAD models using `cadquery` / OpenCascade |
| 🔺 Geometry Tessellation | ✅ Done | Convert B-Rep into triangle mesh using `trimesh` |
| 🧭 Mold Direction Analysis | ✅ Done | Evaluate face normals and select the best pull vector |
| 🔍 Core/Cavity Classification | ✅ Done | Classify faces relative to mold direction |
| ✂️ Parting Line Detection | ✅ Done | Find sign-change adjacency edges and filter for parting line candidates |
| 📊 DfM Scoring | ✅ Done | Compute safe, warning, and undercut face ratios |
| 🖥️ Visualization | ✅ Done | Export interactive Plotly HTML report |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- `pip`

### Installation

```powershell
cd Bosch_DFDM
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the analysis

```powershell
python IDEA1.py
```

The script loads `Part1.stp` by default. To analyze a different STEP file, update the `STEP_FILE` variable at the top of `IDEA1.py`.

---

## 🧪 Output

After execution, the repository produces:

- `Part1_DfM_Result.html` — interactive 3D visualization with mesh, parting lines, and core/cavity coloring
- Console summary including:
  - selected mold direction
  - DfM score
  - safe / warning / undercut face counts
  - outer and inner parting loop statistics

---

## 🧠 How it works

1. Load STEP file using `cadquery`
2. Tessellate the model into vertices and triangular faces
3. Compute face normals and evaluate candidate mold pull directions
4. Classify faces as core or cavity relative to the chosen pull direction
5. Detect parting edges via sign changes between adjacent face orientations
6. Filter parting edges by perpendicularity to the mold pull axis
7. Group parting loops into outer boundary and inner feature loops
8. Render an interactive Plotly visualization and export it to HTML

---

## 📁 Project Files

- `IDEA1.py` — main analysis script
- `Part1.stp` — sample CAD model
- `Part1_DfM_Result.html` — generated visualization
- `requirements.txt` — Python dependencies

---

## 🧩 Technology Stack

- Python 3.11+
- cadquery / OpenCascade (OCP) for STEP import
- trimesh for mesh processing and adjacency analysis
- numpy for geometry math
- scipy for spatial operations
- networkx for graph-based loop detection
- plotly for interactive 3D visualization

---

## 🔧 Notes

- The current implementation is script-based and focused on core DfM analysis for a single STEP file.
- The parting line detection logic is driven by sign-change adjacency and a perpendicularity filter to the selected mold direction.
- Outer and inner parting loops are separated using a radius heuristic.

---

## 📄 License

Developed for the **Bosch DfM Agent Hackathon**. This repository is provided as-is for DfM analysis experimentation and visualization.
