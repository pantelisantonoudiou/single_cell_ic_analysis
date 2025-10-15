# -*- coding: utf-8 -*-
# =============================================================================
#                                 Imports
# =============================================================================

# =============================================================================
# =============================================================================


# if __name__ == '__main__':


# Built-in
import os
import json
from pathlib import Path
from datetime import datetime

# Scientific and plotting
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Custom / Local (unchanged)
from plots_with_stats import group_comparison_plot
from get_index import StimEventIndexer
from batch_process import BatchProcess
from io_secondary_properties import (
    get_basic_properties,
    get_io_properties,
    get_waveform_properties
)

# Plot parameters
mpl.rcParams.update({'font.size': 14})
sns.set(style="whitegrid", font_scale=1.5)

print('All Imports Loaded')

# ================= USER SETTINGS =================
main_path  = r"R:\Kenny\data for analysis\projection-specific resonance\sch"                    # Root path for data
stim_types = ['sch']  #['io', 'rh', 'ch', 'sch']                                                  # Stimulation protocols
analyzed_path    = r'R:\Kenny\data for analysis\projection-specific resonance\sch\analyzed'     # Folder with analyzed data
data_ch    = 0                                                                          # Voltage channel index
stim_ch    = 1                                                                          # Stimulus/current channel index
sep        = '_'                                                                        # Folder name delimiter
output_csv = os.path.join(main_path, 'index.csv')                                       # Index save location
firing_rate_threshold = 60                                                              # Hz, threshold for excluding fast spiking cells
stim_correction = 1000                                                                  # to make stim to pA
data_correction = 100                                                                  # to make data to mV
spike_detection_threshold = 35                                                          # spike detection prominence
njobs       = 1                                                                         # Number of parallel jobs
# ================================================

# Figure/Export roots (added)
FIG_ROOT = Path(analyzed_path) / 'figures'
EXPORT_ROOT = Path(analyzed_path) / 'exports'
FIG_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

# First-run flag
FIRST_RUN_FLAG = Path(analyzed_path) / '.first_run_complete'


def _ensure_section(section: str) -> Path:
    p = FIG_ROOT / section
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_plot_and_data(section: str, name: str, df: pd.DataFrame, fig: plt.Figure, params: dict | None = None):
    sec = _ensure_section(section)
    base = sec / name
    # CSV of the plotted data
    if df is not None:
        df.to_csv(f"{base}.csv", index=False)
    # Meta
    if params:
        with open(f"{base}.meta.json", "w") as f:
            json.dump(params, f, indent=2, default=str)
    # Figures
    fig.savefig(f"{base}.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{base}.svg", bbox_inches='tight')
    # plt.close(fig)


def stamp_run_metadata():
    meta = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'main_path': str(main_path),
        'analyzed_path': str(analyzed_path),
        'stim_types': stim_types,
        'stim_correction': stim_correction,
        'spike_detection_threshold': spike_detection_threshold,
    }
    with open(EXPORT_ROOT / 'run_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
        
indexer = StimEventIndexer(main_path, stim_types, data_ch, stim_ch, sep)
event_df = indexer.build_event_index()
event_df.to_csv(output_csv, index=False)

# Summary
total_files  = event_df['file_name'].nunique() if not event_df.empty else 0
total_events = len(event_df)
print(f"📁 Indexed {total_files} files, found {total_events} stim events.")
# try:
#     display(event_df)
# except Exception:
#     print(event_df.head())


index_csv   = os.path.join(main_path, "index.csv")
analyzed_dir = os.path.join(main_path, 'analyzed')

idx_df = pd.read_csv(index_csv)
processor = BatchProcess(
   main_path, idx_df,
   njobs=njobs,
   stim_correction=stim_correction,
   data_correction=data_correction,
   prominence=spike_detection_threshold
)
processor.run_all(analyzed_dir, stim_types)