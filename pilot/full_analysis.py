#!/usr/bin/env python
# coding: utf-8

# 
# ### Load Imports ###

# In[ ]:


#### Load Imports ###
## include instructiosn how to instal all packages
# Export all plots to a folder as svg and png, for each plot have an associated data file that the user can use to calculate stats
## Add more notes throughout the notebook
# The first time run each cell separately, then after that you can run all at once
# In section 7, create melted vatplot where columns are variables
# data contract: Data index is created from folder names, _ separate conditions
# one file per cell
# file can have more than one block
# the data channel is 0, the stim channel is 1 across all files
# mebrane potential is in mV, current in pA, time in ms, resistance in MOhm, capacitance in pF, conductance in nS
# comments are input-output: io_start & io_stop, rheobase ramp: rh_start & rh_stop, chirp stimulus (subthreshold): ch_start & ch_stop
# short-chirp stimulus (suprathreshold): sch_start & sch_stop
# spike detection prominence is 25 for all cells, stim correction is 1000 for all cells


# Built-in
import os

# Scientific and plotting
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning / Stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Custom / Local
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

print('All Import Loaded')


# ### 1. Create index file with all cells and file properties
# 

# In[ ]:


# ================= USER SETTINGS =================
main_path  = r"R:\Pantelis\for analysis\new patch data for analysis"                    # Root path for data
stim_types = ['io', 'rh', 'ch', 'sch']                                                  # Stimulation protocols
analyzed_path    = r'R:\Pantelis\for analysis\new patch data for analysis\analyzed'     # Path to folder with analyzed data
data_ch    = 0                                                                          # Voltage channel index
stim_ch    = 1                                                                          # Stimulus/current channel index
sep        = '_'                                                                        # Folder name delimiter
output_csv = os.path.join(main_path, 'index.csv')                                       # Save location
firing_rate_threshold = 60                                                              # Hz, threshold for excluding fast spiking cells
stim_correction = 1000 # to make stim to pA
spike_detection_threshold = 25 # to make spike detection prominence
# ================================================

# Build event index
indexer = StimEventIndexer(main_path, stim_types, data_ch, stim_ch, sep)
event_df = indexer.build_event_index()
event_df.to_csv(output_csv, index=False)

# Summary
total_files  = event_df['file_name'].nunique() if not event_df.empty else 0
total_events = len(event_df)
print(f"📁 Indexed {total_files} files, found {total_events} stim events.")
display(event_df)


# ### 2. Extract properties for all four protocols: ['io', 'rh', 'ch', 'sch']

# In[ ]:


# === USER SETTINGS ===
index_csv   = os.path.join(main_path, "index.csv")
analyzed_dir = os.path.join(main_path, 'analyzed')
njobs       = 1
# =====================

idx_df = pd.read_csv(index_csv)
processor = BatchProcess(
   main_path, idx_df,
   njobs=njobs,
   stim_correction=stim_correction,
   prominence=spike_detection_threshold
)
processor.run_all(analyzed_dir, stim_types)


# ### 3. Extract IO secondary properties (ie, waveform metrics and i-o slope, percent firing rate etc)

# In[ ]:


# ================= USER CONFIGURATION =================
stim_type    = 'io'
group_cols   = ['treatment', 'cell_id']
# =====================================================

# ===== Load extracted features =====
basic_path = os.path.join(analyzed_path, f"{stim_type}_basic/features.csv")
wave_path  = os.path.join(analyzed_path, f"{stim_type}_wave/features.csv")
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
io_properties = get_io_properties(plot_io, group_cols, show_plot=False)

# ===== Waveform properties (AP shape) =====
df_wave = df_wave.dropna(subset=['mV', 'time'])
waveform_properties = get_waveform_properties(df_wave, group_cols, show_plot=False)

# ===== Save outputs separately ====
summary_basic_io = io_properties.merge(basic_properties, on=group_cols, how='left')
summary_basic_io.to_csv(os.path.join(analyzed_path, f'{stim_type}_basic', 'summary_io.csv'), index=False)
waveform_properties.to_csv(os.path.join(analyzed_path, f'{stim_type}_wave', 'summary_waveform.csv'), index=False)

# ===== Save filtered feature tables without overwriting original =====
df_basic.to_csv(os.path.join(analyzed_path, f'{stim_type}_basic', 'features_filtered.csv'), index=False)
df_wave.to_csv(os.path.join(analyzed_path, f'{stim_type}_wave', 'features_filtered.csv'), index=False)

print("✅ Property extraction complete.")


# ### 4. Find cells with bad responses and high firing rate to exclude

# In[ ]:


# USER CONFIGURATION
group_col = 'treatment'
palette = ['#1f77b4', '#ff7f0e']

# Load cell metadata
path_basic_io = os.path.join(analyzed_path, 'io_basic', 'summary_io.csv')
df_basic_io = pd.read_csv(path_basic_io)

# Combine
cells_with_bad_responses = set([]) # placeholder if we want to manually add any cells to exclude
# Identify fast spiking cells for exclusion
fast_spiking = set(df_basic_io[df_basic_io['max_firing_rate'] > firing_rate_threshold]['cell_id'])
to_exclude = cells_with_bad_responses.union(fast_spiking)

# Summary per treatment
print(f"🔍 Found {len(cells_with_bad_responses)} cells with bad responses.")
print(f"⚡ Found {len(fast_spiking)} fast spiking cells.")
print(f"❌ Total cells to exclude: {len(to_exclude)}")

print("\n📊 Cell counts by treatment BEFORE exclusion:")
print(df_basic_io.groupby('treatment')['cell_id'].nunique())

remaining = df_basic_io[~df_basic_io['cell_id'].isin(to_exclude)]
print("\n✅ Remaining cells per treatment AFTER exclusion:")
print(remaining.groupby('treatment')['cell_id'].nunique())


# ### 5. I-O Plot

# In[ ]:


# ==== PLOT: IO Spike Frequency vs Amp ====
df_basic = pd.read_csv(os.path.join(analyzed_path, 'io_basic', 'features_filtered.csv'))
df_basic = df_basic[~df_basic['cell_id'].isin(to_exclude)]
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

# ==== PLOT: Rheobase (RH) ====
df_rh = pd.read_csv(os.path.join(analyzed_path, 'rh', 'features.csv'))
df_rh = df_rh.rename(columns={'file_name': 'cell_id', 'condition_0': 'treatment'})
df_rh = df_rh[~df_rh['cell_id'].isin(to_exclude)]
plt.figure(figsize=(6, 5))
sns.barplot(data=df_rh, x='treatment', y='rheobase', palette='pastel', errorbar='se')
sns.stripplot(data=df_rh, x='treatment', y='rheobase', color='black', jitter=True, alpha=0.6)
plt.title('Rheobase by Treatment (RH)')



# ### 6. AP Waveform (Standard Deviation)

# In[ ]:


df_wave  = pd.read_csv(os.path.join(analyzed_path, 'io_wave', 'features_filtered.csv'))
df_wave = df_wave[~df_wave['cell_id'].isin(to_exclude)]
# ==== PLOT: IO Waveform Trace ====
sns.relplot(
    data=df_wave,
    x='time', y='mV',
    hue='treatment', kind='line',
    estimator=np.mean, errorbar='sd',
    palette='tab10',
    height=5, aspect=1.3
)
plt.title('IO: Averaged Waveform by Treatment')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')


# ### 7. IO and Waveform Properties - Summary Plots

# In[ ]:


# Re-load to ensure no prior modification affects the input
import warnings
warnings.filterwarnings('ignore')
df_basic_io = pd.read_csv(os.path.join(analyzed_path, 'io_basic', 'summary_io.csv'))
df_waveform = pd.read_csv(os.path.join(analyzed_path, 'io_wave', 'summary_waveform.csv'))

# Apply exclusions
df_basic_io = df_basic_io[~df_basic_io['cell_id'].isin(to_exclude)]
df_waveform = df_waveform[~df_waveform['cell_id'].isin(to_exclude)]

# Define column sets
basic_io_cols = [
    'fr_at_20_percent_input', 'fr_at_40_percent_input', 'fr_at_60_percent_input',
    'fr_at_80_percent_input', 'fr_at_max_input', 'i_amp_at_half_max_fr',
    'input_resistance', 'resting_membrane_potential', 'max_firing_rate',
    'rheobase', 'io_slope'
]

waveform_cols = [
    'ap_peak', 'threshold', 'ahp', 'peak_to_trough', 'rise_time', 'half_width'
]

## add seaborn plots for 3 categories: basic io, waveform, rh
# melt and plot all variables in basic_io_cols and waveform_cols

sns.catplot(
    data=df_basic_io,
    x='treatment', y='io_slope',
    kind='bar', errorbar='se', palette='pastel',
    height=5, aspect=1
)


# # Run and plot
# print("📊 Analyzing IO features...")
# res_basic_io = group_comparison_plot(
#     df_basic_io,
#     group_column='treatment',
#     dependent_variables=basic_io_cols,
#     palette=palette,
#     n_cols=4
# )

# print("📊 Analyzing waveform features...")
# res_waveform = group_comparison_plot(
#     df_waveform,
#     group_column='treatment',
#     dependent_variables=waveform_cols,
#     palette=palette,
#     n_cols=3
# )


# ### 8. PCA - Logistic regression

# In[ ]:


# ==== COMBINE, PCA + GMM CLUSTERING & PLOT ====
# 1) merge on cell_id & treatment
df_merged = pd.merge(
    df_basic_io, df_waveform,
    on=['cell_id','treatment'], how='inner'
)
df_merged = df_merged.fillna(df_merged.median(numeric_only=True))

# 2) pick features and standardize
features = basic_io_cols + waveform_cols
X = df_merged[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_merged[features].values)
y = df_merged[group_col].values

# 2. PCA to 2D for visualization & classification
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)

# 3. Stratified K-Fold Logistic Regression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
scores = []

for train_idx, test_idx in skf.split(X_pca, y):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)

print(f"✅ Mean CV Accuracy: {np.mean(scores):.3f}")
print(f"📄 Fold Accuracies:  {np.round(scores, 3)}")

# 4. Final model on full PCA data
log_reg.fit(X_pca, y)

# 5. Decision boundary plot
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = log_reg.predict_proba(grid)[:, 1].reshape(xx.shape)

# 6. Scatter + decision boundary
fig, ax = plt.subplots(figsize=(7, 7))
for tr, col in zip(np.unique(y), palette):
    mask = y == tr
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=tr, c=col, edgecolor='k', s=50)

ax.contour(xx, yy, Z, levels=[0.5], linestyles='--', colors='gray')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_title(f'PCA + Logistic Regression\n5-Fold CV Accuracy: {np.mean(scores):.2f}')
ax.legend(title=group_col)
plt.tight_layout()
plt.show()


# ### 9. Chrip (subthreshold resonance) and Short Chirp (suprathreshold spike transfer) normalize and break into bands

# In[ ]:


# === Setup ===
stim_types = ['ch', 'sch']
metrics = {'ch': 'impedance', 'sch': 'spike_count'}
file_paths = {s: os.path.join(analyzed_path, s, 'features.csv') for s in stim_types}
freq_bins = (3, 6, 12, 15, 30, 60)

# === Requires `to_exclude` to be defined already in the notebook ===

# === Delta function ===
def compute_deltas(group, y):
    bp = group.set_index('freq_bins')[y]
    return pd.Series({
        'delta_3-6__6-12':   bp.get(pd.Interval(3, 6),  np.nan) - bp.get(pd.Interval(6, 12), np.nan),
        'delta_15-30__3-6':  bp.get(pd.Interval(15, 30), np.nan) - bp.get(pd.Interval(3, 6), np.nan),
        'delta_15-30__6-12': bp.get(pd.Interval(15, 30), np.nan) - bp.get(pd.Interval(6, 12), np.nan),
    })

# === Storage containers ===
dfs = {}                 # full df with raw and norm columns
dfs_binned_raw = {}      # grouped raw
dfs_binned_norm = {}     # grouped norm
deltas_raw = {}
deltas_norm = {}

# === Process CH and SCH ===
for stim in stim_types:
    metric = metrics[stim]
    
    # Load and clean
    df = pd.read_csv(file_paths[stim])
    df = df.rename(columns={'file_name': 'cell_id', 'condition_0': 'treatment'})
    df = df[~df['cell_id'].isin(to_exclude)].copy()

    # Remove zero-spike cells if needed
    if metric == 'spike_count':
        total = df.groupby(['cell_id', 'treatment'])[metric].transform('sum')
        df = df[total > 0]

    # Normalize
    total = df.groupby(['cell_id', 'treatment'])[metric].transform('sum')
    norm_col = f'norm_{metric}'
    df[norm_col] = df[metric] / total

    # 1) Check sums of norm values per cell×treatment
    check = (
        df
        .groupby(['cell_id', 'treatment'])[norm_col]
        .sum()
        .reset_index(name='sum_norm')
    )

    # 2) Print summaries: should all be (approximately) 1
    print(f"--- {stim.upper()} normalization check ---")
    print(f"Min sum: {check['sum_norm'].min():.4f}, Max sum: {check['sum_norm'].max():.4f}")
    print(f"Any deviating from 1? {((check['sum_norm'] < 0.999) | (check['sum_norm'] > 1.001)).any()}")
    print(check.head(), "\n")

    # Bin frequencies
    df['freq_bins'] = pd.cut(df['freq'], freq_bins)

    # Save full df
    dfs[stim] = df

    # === Binned RAW ===
    df_binned_raw = (
        df.groupby(['freq_bins', 'cell_id', 'treatment'], observed=True)[metric]
        .mean().reset_index()
    )
    dfs_binned_raw[stim] = df_binned_raw

    deltas_raw[stim] = (
        df_binned_raw.groupby(['cell_id', 'treatment'])
        .apply(compute_deltas, y=metric)
        .reset_index()
        .melt(id_vars=['cell_id', 'treatment'], var_name='delta_type', value_name=metric)
    )

    # === Binned NORM ===
    df_binned_norm = (
        df.groupby(['freq_bins', 'cell_id', 'treatment'], observed=True)[norm_col]
        .mean().reset_index()
    )
    dfs_binned_norm[stim] = df_binned_norm

    deltas_norm[stim] = (
        df_binned_norm.groupby(['cell_id', 'treatment'])
        .apply(compute_deltas, y=norm_col)
        .reset_index()
        .melt(id_vars=['cell_id', 'treatment'], var_name='delta_type', value_name=metric)
    )


# In[ ]:


for stim in stim_types:
    metric = metrics[stim]
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=dfs[stim], x='freq', y=metric, hue='treatment', errorbar='se', marker='o')
    plt.title(f'{stim.upper()}: Raw {metric.capitalize()} vs Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()


# ### 10. Plot Normalized (normalized to sum impedance or spike counts per cell)

# In[ ]:


for stim in stim_types:
    norm_col = f'norm_{metrics[stim]}'
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=dfs[stim], x='freq', y=norm_col, hue='treatment', errorbar='se', marker='o')
    plt.title(f'{stim.upper()}: Normalized {metrics[stim].capitalize()} vs Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(norm_col)
    plt.tight_layout()
    plt.show()


# ### 11. Divide to Frequency Bands

# In[ ]:


for stim in stim_types:
    metric = metrics[stim]
    plt.figure(figsize=(7, 5))
    sns.barplot(data=dfs_binned_raw[stim], x='freq_bins', y=metric, hue='treatment', errorbar='se')
    plt.title(f'{stim.upper()}: Binned Raw {metric.capitalize()}')
    plt.xlabel('Frequency Band (Hz)')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

for stim in stim_types:
    norm_col = f'norm_{metrics[stim]}'
    plt.figure(figsize=(7, 5))
    sns.barplot(data=dfs_binned_norm[stim], x='freq_bins', y=norm_col, hue='treatment', errorbar='se')
    plt.title(f'{stim.upper()}: Binned Normalized {metrics[stim].capitalize()}')
    plt.xlabel('Frequency Band (Hz)')
    plt.ylabel(norm_col)
    plt.tight_layout()
    plt.show()


# ### 12. Deltas

# In[ ]:


for stim in stim_types:
    metric = metrics[stim]
    plt.figure(figsize=(7.5, 5))
    sns.barplot(data=deltas_raw[stim], x='delta_type', y=metric, hue='treatment', errorbar='se')
    plt.title(f'{stim.upper()}: Δs on Raw {metric.capitalize()}')
    plt.xlabel('Delta Type')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

for stim in stim_types:
    metric = metrics[stim]
    plt.figure(figsize=(7.5, 5))
    sns.barplot(data=deltas_norm[stim], x='delta_type', y=metric, hue='treatment', errorbar='se')
    plt.title(f'{stim.upper()}: Δs on Normalized {metric.capitalize()}')
    plt.xlabel('Delta Type')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

