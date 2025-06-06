# -*- coding: utf-8 -*-
# =============================================================================
#                                 Imports
# =============================================================================
import os
import pandas as pd
from verify_gui import matplotGui 
import matplotlib.pyplot as plt
# =============================================================================
# =============================================================================

def apply_thresholds_to_stim_index(event_index_path, threshold_df, output_path):
    """
    Merge verified thresholds into stim_event_index and save to new CSV.

    Parameters
    ----------
    event_index_path : str
        Path to original stim_event_index.csv.
    threshold_df : pd.DataFrame
        Contains ['file_name', 'threshold'] for each file.
    output_path : str
        Path to write the enriched CSV.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame with a 'threshold' column for every event row.
    """
    event_df = pd.read_csv(event_index_path)

    # Merge thresholds into full stim event index
    merged_df = event_df.merge(threshold_df, on='file_name', how='left')

    # Warn if any rows are missing a threshold
    missing = merged_df[merged_df['threshold'].isna()]
    if not missing.empty:
        print("⚠️ Warning: Missing threshold for files:")
        print(missing['file_name'].unique())

    # Save to output
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Saved enriched stim index to: {output_path}")
    return merged_df


if __name__ == '__main__':

    ### ----------- USER SETTINGS ------------- ###
    main_path = r"R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys"
    stim_event_index_file = 'index.csv'
    output_file = 'index_verified.csv'
    data_ch = 0
    default_threshold = 30
    ### ---------------------------------------- ###

    # Load full stim index
    stim_df = pd.read_csv(os.path.join(main_path, stim_event_index_file))

    # Prepare GUI DataFrame (one io row per file)
    io_first = (
        stim_df[stim_df['stim_type'] == 'io']
        .sort_values(['file_name', 'block', 'start_sample'])
        .groupby('file_name', as_index=False)
        .first()
    )

    gui_df = io_first[[
        'full_path', 'file_name', 'stim_type', 'block', 'start_sample', 'stop_sample'
    ]].copy()
    gui_df['threshold'] = default_threshold
    gui_df['accepted'] = -1

    # Launch interactive GUI
    gui = matplotGui(gui_df, data_ch=data_ch, prominence=default_threshold)
    plt.show()
    verified_df = gui.get_result()

    # Extract verified thresholds
    threshold_df = verified_df[['file_name', 'threshold']].drop_duplicates()

    # Merge thresholds into full stim index
    enriched_df = stim_df.merge(threshold_df, on='file_name', how='left')

    # Save enriched index to file
    output_path = os.path.join(main_path, output_file)
    enriched_df.to_csv(output_path, index=False)
    print(f"\n🎉 Threshold annotation complete. Total rows: {len(enriched_df)}")
    print(f"✅ Saved to: {output_path}")
