# -*- coding: utf-8 -*-
# =============================================================================
#                                 Imports
# =============================================================================
import os
import pandas as pd
from plots_with_stats import group_comparison_plot
# =============================================================================

if __name__ == '__main__':

    # ========== USER CONFIGURATION ==========
    main_path  = r'R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\analyzed'
    group_col = 'treatment'
    palette = ['#1f77b4', '#ff7f0e'] 
    n_cols    = 4
    # ========================================

    # ========== Load Data ==========
    path_basic_io  = os.path.join(main_path, 'io_basic', 'summary_io.csv')
    path_waveform  = os.path.join(main_path, 'io_wave',  'summary_waveform.csv')

    df_basic_io    = pd.read_csv(path_basic_io)
    df_waveform    = pd.read_csv(path_waveform)

    # ========== Define Columns ==========
    basic_io_cols = [
        'fr_at_20_percent_input', 'fr_at_40_percent_input', 'fr_at_60_percent_input',
        'fr_at_80_percent_input', 'fr_at_max_input', 'i_amp_at_half_max_fr',
        'input_resistance', 'resting_membrane_potential', 'max_firing_rate',
        'rheobase', 'io_slope'
    ]

    waveform_cols = [
        'ap_peak', 'threshold', 'ahp', 'peak_to_trough', 'rise_time', 'half_width'
    ]

    # ========== Plot and Stats ==========
    print("📊 Analyzing Basic/IO features...")
    res_basic_io = group_comparison_plot(
        df_basic_io,
        group_column=group_col,
        dependent_variables=basic_io_cols,
        palette=palette,
        n_cols=n_cols
    )

    print("📊 Analyzing Waveform features...")
    res_waveform = group_comparison_plot(
        df_waveform,
        group_column=group_col,
        dependent_variables=waveform_cols,
        palette=palette,
        n_cols=3
    )

    # ========== Save Results ==========
    res_basic_io.to_csv(os.path.join(main_path, 'stats_io_basic_io.csv'), index=False)
    res_waveform.to_csv(os.path.join(main_path, 'stats_io_waveform.csv'), index=False)

    print("✅ Comparison plots and stats saved.")
