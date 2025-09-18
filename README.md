# Single‑cell Intrinsic Conductance (IC) Analysis — Main Pipeline 🧠

> Tools and notebooks for analyzing single‑cell intrinsic electrophysiology recordings (IO / RH / CH / SCH), computing features, and exporting publication‑quality figures **with paired CSV data**.

---

## TL;DR 🚀

* **Install**: `pip install numpy pandas matplotlib seaborn scikit-learn`
* **Data contract**: one folder **per cell**, file may contain **multiple blocks**; channels fixed (**Vm=0, Stim=1**); units **mV/pA/ms/MΩ/pF/nS**; required tags: `io_start/io_stop`, `rh_start/rh_stop`, `ch_start/ch_stop`, `sch_start/sch_stop`; **prominence=25**, **stim\_correction=1000**.
* **Run**: open **`./main_pipeline.ipynb`**, run the first time **cell‑by‑cell**, then **Run All** afterwards.
* **Outputs**: every plot saved as **SVG + PNG** with an **associated CSV** of the plotted data, under `analyzed/figures/<section>/` and `analyzed/exports/`.

---

## What this pipeline does 📈

This pipeline indexes your single‑cell recordings, runs protocol‑specific analyses for:

* **IO (Input–Output)**: F–I curves, spike counts/frequency, rheobase‑related properties.
* **RH (Rheobase ramp)**: rheobase current per cell.
* **CH (Chirp, subthreshold)**: impedance vs frequency and peak power.
* **SCH (Short‑Chirp, suprathreshold)**: spike transfer vs frequency.

It then:

* computes **secondary properties** (e.g., IO slope, F–I landmarks; AP waveform metrics),
* supports **QC/exclusion** (e.g., fast spikers),
* exports **summary tables**,
* and **saves every figure** as **PNG + SVG** **plus a CSV** with the exact data used to draw the plot.

---

## Install (one‑liner, no environments) 🔧

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

> The pipeline imports local modules kept in this repo (e.g., `plots_with_stats.py`, `get_index.py`, `batch_process.py`, `io_secondary_properties.py`). Make sure these files are present alongside the notebook/scripts.

---

## Data contract (important) 🧾

* **Index & conditions**: the **folder name** is used to create the data index. The underscore `_` **separates conditions**.

  * Example: `CellA_ctrl_drug1` → `cell_id=CellA`, conditions: `ctrl`, `drug1`.
* **One file per cell**, but the file **can contain multiple blocks**.
* **Channels** (fixed for all files):

  * **data channel (Vm)** = **0**
  * **stim channel (current)** = **1**
* **Units**:

  * membrane potential **mV**, current **pA**, time **ms**, resistance **MΩ**, capacitance **pF**, conductance **nS**
* **Comment tags** (epoch boundaries):

  * IO: `io_start` & `io_stop`
  * Rheobase ramp: `rh_start` & `rh_stop`
  * Chirp (subthreshold): `ch_start` & `ch_stop`
  * Short‑chirp (suprathreshold): `sch_start` & `sch_stop`
* **Spike detection**: prominence **= 25** for all cells.
* **Stimulus correction**: factor **= 1000** applied to all stim traces.

> Keep the **channel order and units** consistent across files to avoid downstream conversion issues.

---

## Folder structure & paths 🗂️

You control paths via the top‑level settings in the notebook (e.g., `main_path`, `analyzed_path`). A typical layout:

```
<repo_root>/
├─ main_pipeline.ipynb          # ← entry point
├─ batch_process.py
├─ get_index.py
├─ io_secondary_properties.py
├─ plots_with_stats.py
├─ ...other helpers...
└─ <main_path>/                 # your raw/processed cell data root
   ├─ CellA_ctrl_drug1/
   │  └─ CellA.csv (or .npz)
   ├─ CellB_ctrl/
   │  └─ CellB.csv
   └─ analyzed/                 # (created by the pipeline)
      ├─ io_basic/              # per‑protocol features & summaries
      ├─ io_wave/
      ├─ rh/
      ├─ ch/
      ├─ sch/
      ├─ figures/               # all exported figures
      │  └─ <section>/<name>.png, .svg, .csv, .meta.json
      └─ exports/               # run metadata, exclusion lists, combined tables
         └─ run_metadata.json, excluded_cells.csv, ...
```

---

## How to run the main notebook ▶️

1. Launch Jupyter and open **`./main_pipeline.ipynb`** (at the repo root).
2. **First run**: execute **top‑to‑bottom cell‑by‑cell** to create output folders, index files, and validate paths/settings.
3. After the first run, you can **Run All**.
4. Inspect outputs under `analyzed/figures/` (PNG+SVG+CSV per plot) and `analyzed/exports/`.

### Notes

* The indexing step writes `index.csv` under `main_path` and prints a summary of discovered events.
* QC step writes `analyzed/exports/excluded_cells.csv` describing excluded cell IDs (e.g., fast spikers).

---

## What the other scripts do 🧰

* **`get_index.py`** — Builds a stimulus event index from your data root (derives `cell_id` and **conditions from folder names**; applies channel assumptions).
* **`batch_process.py`** — Batch runner that executes analyses over all cells for the specified protocols (`io`, `rh`, `ch`, `sch`), applying **stimulus correction = 1000** and **spike prominence = 25**.
* **`io_secondary_properties.py`** — Computes secondary metrics (e.g., IO slope, firing‑rate landmarks) and AP waveform properties.
* **`plots_with_stats.py`** — Plotting helpers and statistical overlays used by the notebook.
* **`plots/…` or `plot_*.py` (if present)** — Additional figure builders for specific analyses.

> All plotting entry points use the shared exporter so **every figure** includes **SVG + PNG + CSV** (and a small `.meta.json`).

---

## Troubleshooting 🛠️

* **No cells detected**: check `main_path` and that folder names match the contract (underscore‑separated conditions).
* **Missing tags**: ensure each protocol has the proper `*_start`/`*_stop` markers.
* **Incorrect channels**: verify Vm is channel **0** and Stim is channel **1**.
* **Units off**: confirm data are in **mV/pA/ms** and that **stimulus correction = 1000** is applied once.

---

## License 📜

Unless otherwise noted, this project is released under Apache‑2.0 (see `LICENSE`).
