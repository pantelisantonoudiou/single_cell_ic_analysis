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
# =============================================================================

# =============================================================================
#                                 Functions
# =============================================================================
def compute_deltas(group, y):
    """Compute 3 deltas on band-binned metric y."""
    bp = group.set_index('freq_bins')[y]
    return pd.Series({
        'delta_3-6__6-12':   bp.get(pd.Interval(3, 6),  np.nan)
                             - bp.get(pd.Interval(6, 12), np.nan),
        'delta_15-30__3-6':  bp.get(pd.Interval(15, 30), np.nan)
                             - bp.get(pd.Interval(3, 6),   np.nan),
        'delta_15-30__6-12': bp.get(pd.Interval(15, 30), np.nan)
                             - bp.get(pd.Interval(6, 12),  np.nan),
    })

if __name__ == '__main__':
    
    # =============================================================================
    #                                 Config
    # =============================================================================
    main_path   = r'R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\full_dataset\analyzed'
    stim_types  = ('ch', 'sch')
    metrics     = ('impedance', 'spike_count')    # one metric per stim
    file_paths  = {s: os.path.join(main_path, s, 'features.csv')
                   for s in stim_types}
    freq_bins   = (3, 6, 12, 15, 30, 60)
    
    # =============================================================================
    #                        Load → Filter → Normalize → Bin → Δ
    # =============================================================================
    dfs, deltas_dfs = {}, {}
    
    for stim, metric in zip(stim_types, metrics):
        df = pd.read_csv(file_paths[stim])
        df = df.rename(columns={
            'file_name':   'cell_id',
            'condition_0': 'treatment'
        })
        # 1) Exclude zero-spike cells (only for spike_count)
        if metric == 'spike_count':
            tot = df.groupby(['cell_id','treatment'])[metric].transform('sum')
            df = df[tot > 0]
    
        # 2) Normalize into new column norm_<metric>
        norm_col = f'norm_{metric}'
        total = df.groupby(['cell_id','treatment'])[metric].transform('sum')
        df[norm_col] = df[metric] / total
    
        # 3) Bin and average *only* the normalized metric
        df['freq_bins'] = pd.cut(df['freq'], freq_bins)
        df = df.dropna(subset=['freq_bins'])
        df_binned = (
            df
            .groupby(['freq_bins','cell_id','treatment'], observed=True)
            [norm_col]
            .mean()
            .reset_index()
        )
        dfs[stim] = df_binned
    
        # 4) Compute 3 deltas per cell×treatment, then melt into long form
        wide = (
            df_binned
            .groupby(['cell_id','treatment'])
            .apply(compute_deltas, y=norm_col, include_groups=False)
            .reset_index()
        )
        long = wide.melt(
            id_vars=['cell_id','treatment'],
            value_vars=[
                'delta_3-6__6-12',
                'delta_15-30__3-6',
                'delta_15-30__6-12'
            ],
            var_name   = 'delta_type',
            value_name = metric   # the y-column is named after the original metric
        )
        deltas_dfs[stim] = long
    
    # =============================================================================
    #                               Example Plots
    # =============================================================================
    # 1) Normalized metric across freq_bins
    plt.figure(figsize=(7,5))
    sns.barplot(data=dfs['ch'], x='freq_bins', y='norm_impedance',
                hue='treatment', errorbar='se')
    plt.title('CH: Normalized Impedance vs Frequency')
    plt.xlabel('Freq (Hz)'); plt.ylabel('norm_impedance')
    plt.tight_layout()
    
    plt.figure(figsize=(7,5))
    sns.barplot(data=dfs['sch'], x='freq_bins', y='norm_spike_count',
                hue='treatment', errorbar='se')
    plt.title('SCH: Normalized Spike Count vs Frequency')
    plt.xlabel('Freq (Hz)'); plt.ylabel('norm_spike_count')
    plt.tight_layout()
    
    # 2) All 3 deltas, grouped by delta_type
    plt.figure(figsize=(8,5))
    sns.barplot(data=deltas_dfs['ch'], x='delta_type', y='impedance',
                hue='treatment', errorbar='se')
    plt.title('CH: Δ’s on Normalized Impedance')
    plt.xlabel('Delta Type'); plt.ylabel('impedance')
    plt.tight_layout()
    
    plt.figure(figsize=(8,5))
    sns.barplot(data=deltas_dfs['sch'], x='delta_type', y='spike_count',
                hue='treatment', errorbar='se')
    plt.title('SCH: Δ’s on Normalized Spike Count')
    plt.xlabel('Delta Type'); plt.ylabel('spike_count')
    plt.tight_layout()
    
    plt.show()
