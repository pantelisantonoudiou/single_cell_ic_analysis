# single\_cell\_ic\_analysis 🧪

Tools and notebooks for analyzing single-cell intrinsic electrophysiology recordings (current clamp and resonance), extracting input–output (I/O) properties, and generating publication-quality figures.

> **Status:** research code, evolving.
> **License:** Apache-2.0

---

## 📂 Contents

* **Batch & pipelines**

  * `batch_process.py` — run end-to-end analyses over many cells/sweeps.
  * `full_analysis.ipynb` — interactive, step-by-step analysis in Jupyter.
  * `full_analysis_script..py` — scripted equivalent of the notebook.

* **Core analysis**

  * `current_clamp.py` — helpers for current-clamp traces (spike detection, F-I, rheobase, etc.).
  * `get_io_properties.py` — compute primary I/O features per cell/sweep.
  * `io_secondary_properties.py` — derived/secondary metrics and summaries.
  * `plot_primary.py`, `plot_resonance.py`, `plot_io_properties.py`, `plots_with_stats.py` — figure generation and stats overlays.

* **Utilities**

  * `get_index.py` — indexing/lookup of recordings/cells across datasets.
  * `excluded_cells.py` — curated list/logic of cells to omit (QC).

---

## 🚀 Quick start

### 1) Clone or download the repo

```bash
git clone https://github.com/pantelisantonoudiou/single_cell_ic_analysis.git
cd single_cell_ic_analysis
```

### 2) Set up an environment

```bash

Open conda prompt and cd to repository
conda env create -f environment.yml
conda activate single_cell_ic
```

### 3) Run an analysis

**Notebook (interactive):**

```bash
jupyter lab full_analysis.ipynb
```

**Script (batch):**

```bash
# Edit input/output paths at the top of the script.
python batch_process.py
```

Outputs (per-cell CSV summaries, plots) will be written to the paths configured in the script/notebook.

---

## 📊 Data assumptions

* **Recordings:** single-cell current-clamp sweeps (e.g., step protocols) and optional resonance/impedance protocols.
* **Organization:** one folder per experiment type; the repository uses `get_index.py` to locate and enumerate inputs.
* **Exclusions/QC:** `excluded_cells.py` centralizes cells/sweeps to ignore; update this to reproduce specific figures/tables.

---

## 🔬 What the pipeline computes (typical)

* Spike features per sweep (threshold, peak, width, AHP)
* F–I curves & gain, rheobase estimation
* Subthreshold features: input resistance, membrane time constant (τ), sag
* Resonance amplitude/frequency & impedance profiles (if protocol present)
* Per-cell summaries and cohort-level stats/plots

---

## 📁 Project structure

```
single_cell_ic_analysis/
├─ full_analysis.ipynb
├─ full_analysis_script..py
├─ batch_process.py
├─ current_clamp.py
├─ get_io_properties.py
├─ io_secondary_properties.py
├─ plot_primary.py
├─ plot_resonance.py
├─ plot_io_properties.py
├─ plots_with_stats.py
├─ get_index.py
├─ excluded_cells.py
├─ LICENSE
└─ README.md
```

---

## 📜 License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
