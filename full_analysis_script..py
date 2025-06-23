import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from plots_with_stats import group_comparison_plot
import warnings
warnings.filterwarnings('ignore')

# ====== Config ======
analysis_path = r'R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\full_dataset\analyzed'
group_col = 'treatment'
palette = ['#1f77b4', '#ff7f0e']
plot_dir = os.path.join(analysis_path, "plots")
os.makedirs(plot_dir, exist_ok=True)

def savefig(name):
    path = os.path.join(plot_dir, f"{name}.pdf")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")

# ====== Exclusions ======
df_basic_io = pd.read_csv(os.path.join(analysis_path, 'io_basic', 'summary_io.csv'))

bad_ids = ['fctrap5_', 'fctrap7_', 'fctrap8_', 'fctrap13_',
           'exttrap1_', 'exttrap10_', 'exttrap12_', 'exttrap13_', 'exttrap15_']
pattern = '|'.join(map(re.escape, bad_ids))
cells_with_bad_responses = set(df_basic_io[df_basic_io['cell_id'].str.contains(pattern, case=False, na=False)]['cell_id'])
fast_spiking = set(df_basic_io[df_basic_io['max_firing_rate'] > 60]['cell_id'])
to_exclude = cells_with_bad_responses.union(fast_spiking)

# ====== Step 5: IO Plot ======
df_basic = pd.read_csv(os.path.join(analysis_path, 'io_basic', 'features_filtered.csv'))
df_basic = df_basic[~df_basic['cell_id'].isin(to_exclude)]
sns.relplot(data=df_basic, x='amp', y='spike_frequency', hue='treatment',
            kind='line', marker='o', errorbar='se', height=6, aspect=1.5, palette='tab10')
plt.title('IO: Spike Frequency vs Current Amplitude')
plt.xlabel('Current Injection (pA)')
plt.ylabel('Spike Frequency (Hz)')
savefig("io_spike_vs_amp")

df_rh = pd.read_csv(os.path.join(analysis_path, 'rh', 'features.csv'))
df_rh = df_rh.rename(columns={'file_name': 'cell_id', 'condition_0': 'treatment'})
df_rh = df_rh[~df_rh['cell_id'].isin(to_exclude)]
plt.figure(figsize=(8, 6))
sns.barplot(data=df_rh, x='treatment', y='rheobase', palette='pastel', errorbar='se')
sns.stripplot(data=df_rh, x='treatment', y='rheobase', color='black', jitter=True, alpha=0.6)
plt.title('Rheobase by Treatment (RH)')
savefig("rh_rheobase_by_treatment")

# ====== Step 6: Waveform Trace ======
df_wave = pd.read_csv(os.path.join(analysis_path, 'io_wave', 'features_filtered.csv'))
df_wave = df_wave[~df_wave['cell_id'].isin(to_exclude)]
sns.relplot(data=df_wave, x='time', y='mV', hue='treatment',
            kind='line', estimator=np.mean, errorbar='sd', height=6, aspect=1.5, palette='tab10')
plt.title('IO: Averaged Waveform by Treatment')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
savefig("io_waveform_avg")

# ====== Step 7: Summary Features ======
df_basic_io = pd.read_csv(os.path.join(analysis_path, 'io_basic', 'summary_io.csv'))
df_waveform = pd.read_csv(os.path.join(analysis_path, 'io_wave', 'summary_waveform.csv'))
df_basic_io = df_basic_io[~df_basic_io['cell_id'].isin(to_exclude)]
df_waveform = df_waveform[~df_waveform['cell_id'].isin(to_exclude)]

basic_io_cols = [
    'fr_at_20_percent_input', 'fr_at_40_percent_input', 'fr_at_60_percent_input',
    'fr_at_80_percent_input', 'fr_at_max_input', 'i_amp_at_half_max_fr',
    'input_resistance', 'resting_membrane_potential', 'max_firing_rate',
    'rheobase', 'io_slope'
]
waveform_cols = ['ap_peak', 'threshold', 'ahp', 'peak_to_trough', 'rise_time', 'half_width']

_ = group_comparison_plot(df_basic_io, group_column='treatment',
                          dependent_variables=basic_io_cols, palette=palette, n_cols=4)
savefig("summary_io_features")

_ = group_comparison_plot(df_waveform, group_column='treatment',
                          dependent_variables=waveform_cols, palette=palette, n_cols=3)
savefig("summary_waveform_features")

# ====== Step 8: PCA & Logistic Regression ======
features = basic_io_cols + waveform_cols
df_merged = pd.merge(df_basic_io, df_waveform, on=['cell_id', 'treatment'], how='inner')
df_merged = df_merged.fillna(df_merged.median(numeric_only=True))
X = df_merged[features].values
y = df_merged[group_col].values
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = [accuracy_score(y[test], log_reg.fit(X_pca[train], y[train]).predict(X_pca[test]))
          for train, test in cv.split(X_pca, y)]

log_reg.fit(X_pca, y)
xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 500),
                     np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 500))
Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

plt.figure(figsize=(8, 7))
for label, color in zip(np.unique(y), palette):
    mask = y == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, c=color, edgecolor='k', s=60)
plt.contour(xx, yy, Z, levels=[0.5], linestyles='--', colors='gray')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title(f'PCA + Logistic Regression\nMean CV Accuracy: {np.mean(scores):.2f}')
plt.legend(title=group_col)
savefig("pca_logistic_regression")

# ====== Step 9–12: Chirp & SCH Frequency Plots ======
stim_types = ['ch', 'sch']
metrics = {'ch': 'impedance', 'sch': 'spike_count'}
file_paths = {s: os.path.join(analysis_path, s, 'features.csv') for s in stim_types}
freq_bins = (3, 6, 12, 15, 30, 60)

def compute_deltas(group, y):
    bp = group.set_index('freq_bins')[y]
    return pd.Series({
        'delta_3-6__6-12': bp.get(pd.Interval(3, 6), np.nan) - bp.get(pd.Interval(6, 12), np.nan),
        'delta_15-30__3-6': bp.get(pd.Interval(15, 30), np.nan) - bp.get(pd.Interval(3, 6), np.nan),
        'delta_15-30__6-12': bp.get(pd.Interval(15, 30), np.nan) - bp.get(pd.Interval(6, 12), np.nan),
    })

for stim in stim_types:
    metric = metrics[stim]
    df = pd.read_csv(file_paths[stim])
    df = df.rename(columns={'file_name': 'cell_id', 'condition_0': 'treatment'})
    df = df[~df['cell_id'].isin(to_exclude)].copy()
    if metric == 'spike_count':
        total = df.groupby(['cell_id', 'treatment'])[metric].transform('sum')
        df = df[total > 0]
    total = df.groupby(['cell_id', 'treatment'])[metric].transform('sum')
    norm_col = f'norm_{metric}'
    df[norm_col] = df[metric] / total
    df['freq_bins'] = pd.cut(df['freq'], freq_bins)

    # Raw line
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='freq', y=metric, hue='treatment', errorbar='se', marker='o')
    plt.title(f'{stim.upper()}: Raw {metric.capitalize()} vs Frequency')
    savefig(f"{stim}_raw_line")

    # Normalized line
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='freq', y=norm_col, hue='treatment', errorbar='se', marker='o')
    plt.title(f'{stim.upper()}: Normalized {metric.capitalize()} vs Frequency')
    savefig(f"{stim}_norm_line")

    # Binned bar
    df_binned_raw = df.groupby(['freq_bins', 'cell_id', 'treatment'], observed=True)[metric].mean().reset_index()
    df_binned_norm = df.groupby(['freq_bins', 'cell_id', 'treatment'], observed=True)[norm_col].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_binned_raw, x='freq_bins', y=metric, hue='treatment', errorbar='se')
    plt.title(f'{stim.upper()}: Binned Raw {metric.capitalize()}')
    savefig(f"{stim}_binned_raw")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_binned_norm, x='freq_bins', y=norm_col, hue='treatment', errorbar='se')
    plt.title(f'{stim.upper()}: Binned Normalized {metric.capitalize()}')
    savefig(f"{stim}_binned_norm")

    # Deltas
    delta_raw = (
        df_binned_raw.groupby(['cell_id', 'treatment'])
        .apply(compute_deltas, y=metric)
        .reset_index()
        .melt(id_vars=['cell_id', 'treatment'], var_name='delta_type', value_name=metric)
    )
    delta_norm = (
        df_binned_norm.groupby(['cell_id', 'treatment'])
        .apply(compute_deltas, y=norm_col)
        .reset_index()
        .melt(id_vars=['cell_id', 'treatment'], var_name='delta_type', value_name=metric)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=delta_raw, x='delta_type', y=metric, hue='treatment', errorbar='se')
    plt.title(f'{stim.upper()}: Δs on Raw {metric.capitalize()}')
    savefig(f"{stim}_delta_raw")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=delta_norm, x='delta_type', y=metric, hue='treatment', errorbar='se')
    plt.title(f'{stim.upper()}: Δs on Normalized {metric.capitalize()}')
    savefig(f"{stim}_delta_norm")
