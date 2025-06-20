# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from io_secondary_properties import (
    get_basic_properties,
    get_io_properties,
    get_waveform_properties
)

# ================= USER CONFIGURATION =================
main_path    = r'R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\full_dataset\analyzed'
stim_type    = 'io'
group_cols   = ['treatment', 'cell_id']
# =====================================================

# ===== Load extracted features =====
basic_path = os.path.join(main_path, f"{stim_type}_basic/features.csv")
wave_path  = os.path.join(main_path, f"{stim_type}_wave/features.csv")
df_basic   = pd.read_csv(basic_path)
df_wave    = pd.read_csv(wave_path)

# ===== Standardize column names =====
df_basic = df_basic.rename(columns={'file_name': 'cell_id', 'condition_0': 'treatment'})
df_wave  = df_wave.rename(columns={'file_name': 'cell_id', 'condition_0': 'treatment'})

# ===== Filter out non-spiking cells =====
spike_summary = df_basic.groupby('cell_id')['spike_frequency'].max().reset_index()
non_spiking = spike_summary[spike_summary['spike_frequency'] <= 0]['cell_id']
df_basic = df_basic[~df_basic['cell_id'].isin(non_spiking)].reset_index(drop=True)
df_wave  = df_wave[~df_wave['cell_id'].isin(non_spiking)].reset_index(drop=True)

# ===== Basic properties (RMP, Rin) =====
plot_basic = (
    df_basic.groupby(group_cols)[['input_resistance', 'rmp']]
    .mean()
    .reset_index()
)
plot_basic.loc[plot_basic['input_resistance'] < 0, 'input_resistance'] = np.NaN
basic_properties = get_basic_properties(plot_basic, group_cols)

# ===== IO curve properties (freq, slope, rheobase) =====
plot_io = (
    df_basic.groupby(group_cols + ['amp'])[['spike_frequency']]
    .mean()
    .reset_index()
)
plot_io = plot_io[plot_io['amp'] > 0]
io_properties = get_io_properties(plot_io, group_cols, show_plot=True)

# ===== Waveform properties (AP shape) =====
df_wave = df_wave.dropna(subset=['mV', 'time'])
waveform_properties = get_waveform_properties(df_wave, group_cols, show_plot=True)

# ===== Save outputs separately ====
summary_basic_io = io_properties.merge(basic_properties, on=group_cols, how='left')
summary_basic_io.to_csv(os.path.join(main_path, f'{stim_type}_basic', 'summary_io.csv'), index=False)
waveform_properties.to_csv(os.path.join(main_path, f'{stim_type}_wave', 'summary_waveform.csv'), index=False)

# ===== Save filtered feature tables without overwriting original =====
df_basic.to_csv(os.path.join(main_path, f'{stim_type}_basic', 'features_filtered.csv'), index=False)
df_wave.to_csv(os.path.join(main_path, f'{stim_type}_wave', 'features_filtered.csv'), index=False)

print("✅ Property extraction complete.")
