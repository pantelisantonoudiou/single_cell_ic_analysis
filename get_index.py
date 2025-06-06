# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import adi
import pandas as pd
##### ------------------------------------------------------------------- #####


def get_file_data(main_path: str, sep: str = '_') -> pd.DataFrame:
    """
    Scan subdirectories for .adicht files and return a DataFrame of file metadata.

    Each row represents one .adicht file with the following columns:
        - folder_path : name of the subdirectory (lowercased)
        - file        : filename (lowercased)
        - full_path   : absolute path to the file
        - condition   : derived from the first segment of folder name split by `sep`
        - (extra columns): additional segments of the folder name

    Parameters
    ----------
    main_path : str
        Path to the parent directory containing subfolders with .adicht files.
    sep : str, optional
        Delimiter used to split subfolder names into metadata (default: "_").

    Returns
    -------
    pd.DataFrame
        Table containing file metadata and full path for each .adicht file.
    """
    main_path = os.path.normpath(main_path)
    dirs = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]

    df_list = []

    for folder in dirs:
        subfolder_path = os.path.join(main_path, folder)
        filelist = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.adicht')]

        if not filelist:
            continue

        split_labels = folder.split(sep)
        temp_df = pd.DataFrame([split_labels] * len(filelist))
        temp_df.insert(0, 'file', [f.lower() for f in filelist])
        temp_df.insert(0, 'folder_path', folder.lower())
        temp_df.insert(0, 'file_path', [os.path.join(subfolder_path, f) for f in filelist])

        df_list.append(temp_df)

    if not df_list:
        return pd.DataFrame(columns=['folder_path', 'file', 'file_path'])

    file_data = pd.concat(df_list, ignore_index=True)
    file_data.columns = ['file_path', 'folder_path', 'file'] + [f'meta_{i}' for i in range(file_data.shape[1] - 3)]
    file_data['condition'] = file_data['meta_0'] if 'meta_0' in file_data.columns else None
    file_data = file_data.apply(lambda col: col.astype(str).str.lower())

    return file_data


def get_comment_ranges_for_file(
    file_path: str,
    file_name: str,
    stim_types: list,
    data_ch: int,
    stim_ch: int
) -> list:
    """
    Extract stimulation event metadata from a single ADI file.

    Parameters
    ----------
    file_path : str
        Full path to the ADI file.
    file_name : str
        Name of the file (used for tracking).
    stim_types : list of str
        Stimulus types to search for (e.g., ['io', 'rh']).
    data_ch : int
        Channel index for voltage/recording.
    stim_ch : int
        Channel index for stimulation/current.

    Returns
    -------
    list of dict
        One dict per stimulation event.
    """
    try:
        fread = adi.read_file(file_path)
        data_obj = fread.channels[data_ch]
        fs = int(data_obj.fs[0])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

    results = []

    for block_idx, record in enumerate(data_obj.records):
        comments = record.comments
        if not comments:
            continue

        com_texts = [c.text.lower() for c in comments]
        com_ticks = {c.text.lower(): c.tick_position for c in comments}

        for stim_type in stim_types:
            start_comment = f"{stim_type}_start"
            stop_comment = f"{stim_type}_stop"

            if start_comment in com_texts:
                start_sample = com_ticks[start_comment] + 1
                if stim_type == 'io':
                    start_sample = max(1, start_sample - int(fs / 2))

                stop_sample = com_ticks.get(stop_comment, None)

                results.append({
                    'full_path': file_path,
                    'file_name': file_name,
                    'stim_type': stim_type,
                    'block': block_idx,
                    'start_sample': start_sample,
                    'stop_sample': stop_sample,
                    'data_ch': data_ch,
                    'stim_ch': stim_ch,
                    'fs': fs
                })

    return results


def get_comment_ranges(index_df: pd.DataFrame, stim_types: list, data_ch: int, stim_ch: int) -> pd.DataFrame:
    """
    Extract stimulation event metadata from all files in `index_df`.

    Parameters
    ----------
    index_df : pd.DataFrame
        Must contain 'full_path' and 'file' columns.
    stim_types : list of str
        Stim labels to look for (e.g., ['io', 'rh']).
    data_ch : int
        Vm channel index.
    stim_ch : int
        Stim channel index.

    Returns
    -------
    pd.DataFrame
        Rows: stim events
        Columns: metadata including full_path, file, block, sample indices, etc.
    """
    all_results = []

    for _, row in index_df.iterrows():
        file_results = get_comment_ranges_for_file(
            file_path=row['file_path'],
            file_name=row['file'],
            stim_types=stim_types,
            data_ch=data_ch,
            stim_ch=stim_ch
        )
        all_results.extend(file_results)

    if not all_results:
        return pd.DataFrame(columns=[
            'file_path', 'file_name', 'stim_type', 'block',
            'start_sample', 'stop_sample', 'data_ch', 'stim_ch', 'fs'
        ])

    return pd.DataFrame(all_results)


if __name__ == '__main__':
    # === Configuration ===
    main_path = r"R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys"
    stim_types = ['io', 'rh', 'ch', 'sch']
    data_ch = 0
    stim_ch = 1

    # === Step 1: Build File Index ===
    index_df = get_file_data(main_path, sep='_')

    # === Step 2: Get Stim Event Metadata ===
    comment_index_df = get_comment_ranges(index_df, stim_types, data_ch, stim_ch)

    # === Step 3: Join with condition or other metadata ===
    if 'condition' in index_df.columns:
        comment_index_df = comment_index_df.merge(
            index_df[['file', 'condition']], 
            left_on='file_name', right_on='file', 
            how='left'
        ).drop(columns='file')

    # === Step 4: Save to CSV ===
    output_csv = os.path.join(main_path, 'index.csv')
    comment_index_df.to_csv(output_csv, index=False)
    print(f"\n✅ Stim event index saved to: {output_csv}")
