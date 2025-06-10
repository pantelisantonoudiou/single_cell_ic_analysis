# -*- coding: utf-8 -*-
# =============================================================================
#                                 Imports
# =============================================================================
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
sns.set(style="whitegrid", font_scale=1.5)
# =============================================================================

# ==== CONFIG ====
main_path = r'R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\analyzed'
stim_types = ['rh', 'ch', 'sch']
file_paths = {stim: os.path.join(main_path, stim, 'features.csv') for stim in stim_types}

# ==== LOAD STIM-TYPE DATA (RH, CH, SCH) ====
dfs = {}
for stim in stim_types:
    df = pd.read_csv(file_paths[stim])
    df = df.rename(columns={'file_name': 'cell_id', 'condition_0': 'treatment'})
    dfs[stim] = df

# ==== LOAD IO DATA ====
df_basic = pd.read_csv(os.path.join(main_path, 'io_basic', 'features_filtered.csv'))
df_wave  = pd.read_csv(os.path.join(main_path, 'io_wave', 'features_filtered.csv'))

# ==== PLOT: IO Spike Frequency vs Amp ====
sns.relplot(
    data=df_basic,
    x='amp', y='spike_frequency',
    marker='o', hue='treatment',
    kind='line', errorbar='se',
    palette='tab10',
    height=5, aspect=1.3
)
plt.title('IO: Spike Frequency vs Current Amplitude')
plt.xlabel('Current Injection (pA)')
plt.ylabel('Spike Frequency (Hz)')

# ==== PLOT: IO Waveform Trace ====
sns.relplot(
    data=df_wave,
    x='time', y='mV',
    hue='treatment', kind='line',
    estimator=np.mean, errorbar='se',
    palette='tab10',
    height=5, aspect=1.3
)
plt.title('IO: Averaged Waveform by Treatment')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')

# ==== PLOT: Rheobase (RH) ====
plt.figure(figsize=(6, 5))
sns.barplot(data=dfs['rh'], x='treatment', y='rheobase', palette='pastel', errorbar='se')
sns.stripplot(data=dfs['rh'], x='treatment', y='rheobase', color='black', jitter=True, alpha=0.6)
plt.title('Rheobase by Treatment (RH)')

# ==== PLOT: Impedance vs Frequency (CH) ====
plt.figure(figsize=(7, 5))
sns.lineplot(
    data=dfs['ch'],
    x='freq', y='impedance',
    hue='treatment', errorbar='se', marker='o'
)
plt.title('CH: Chirp Stimulus')
plt.xlabel( 'Frequency (Hz)')
plt.ylabel('Impedance')

# ==== PLOT: Spikes vs Stim Frequency (SCH) ====
plt.figure(figsize=(7, 5))
sns.lineplot(
    data=dfs['sch'],
    x='freq', y='spike_count',
    hue='treatment', errorbar='se', marker='o'
)
plt.title('SCH: Spikes vs Stim Frequency')
plt.xlabel('Stim Frequency (Hz)')
plt.ylabel('Spike Count')
