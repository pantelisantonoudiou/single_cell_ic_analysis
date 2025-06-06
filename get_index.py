# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import adi
import pandas as pd
##### ------------------------------------------------------------------- #####


class StimEventIndexer:
    """
    Class to scan `.adicht` files in a directory tree, extract stimulation event metadata,
    and build a unified stim event index.

    Each subfolder under `main_path` is assumed to contain .adicht files. The subfolder's
    name is split by `sep` into multiple condition tokens (e.g., "pv_gfp" → ["pv", "gfp"]).
    These become columns `condition_0`, `condition_1`, etc.

    Parameters
    ----------
    main_path : str
        Path to the parent directory containing subfolders of `.adicht` files.
    stim_types : list of str
        List of stimulation prefixes to search for in the comment texts, e.g. ['io', 'rh'].
    data_ch : int
        Index of the voltage (Vm) channel in the ADI file.
    stim_ch : int
        Index of the stimulus/current channel in the ADI file.
    sep : str, optional
        Delimiter used to split subfolder names into condition tokens (default: "_").

    Attributes
    ----------
    index_df : pd.DataFrame
        DataFrame of all `.adicht` files with columns:
        ['file_path', 'file_name', 'folder_name', 'condition_0', 'condition_1', ...]
    event_df : pd.DataFrame
        DataFrame of all stimulation events with columns:
        ['file_path', 'file_name', 'stim_type', 'block', 'start_sample', 'stop_sample',
         'data_ch', 'stim_ch', 'fs', 'condition_0', 'condition_1', ...]
    """

    # Column name constants
    COL_FILE_PATH   = 'file_path'
    COL_FILE_NAME   = 'file_name'
    COL_FOLDER      = 'folder_name'
    COL_STIM_TYPE   = 'stim_type'
    COL_BLOCK       = 'block'
    COL_START       = 'start_sample'
    COL_STOP        = 'stop_sample'
    COL_DATA_CH     = 'data_ch'
    COL_STIM_CH     = 'stim_ch'
    COL_FS          = 'fs'

    def __init__(self, main_path: str, stim_types: list, data_ch: int, stim_ch: int, sep: str = '_'):
        self.main_path = os.path.normpath(main_path)
        self.stim_types = stim_types
        self.data_ch = data_ch
        self.stim_ch = stim_ch
        self.sep = sep

        # These will be populated by methods
        self.index_df = pd.DataFrame()
        self.event_df = pd.DataFrame()

    def scan_files(self) -> pd.DataFrame:
        """
        Scan all immediate subdirectories of `main_path` for `.adicht` files.
        Split each subfolder's name by `sep` into multiple condition tokens:
            condition_0, condition_1, ..., as many as there are tokens.

        Returns
        -------
        pd.DataFrame
            Columns:
                ['file_path', 'file_name', 'folder_name', 'condition_0', 'condition_1', ...]
        """
        rows = []
        for folder in os.listdir(self.main_path):
            folder_path = os.path.join(self.main_path, folder)
            if not os.path.isdir(folder_path):
                continue

            tokens = folder.lower().split(self.sep)
            meta_cols = {f'condition_{i}': tokens[i] for i in range(len(tokens))}

            for fname in os.listdir(folder_path):
                if not fname.lower().endswith('.adicht'):
                    continue
                row = {
                    self.COL_FILE_PATH: os.path.join(folder_path, fname),
                    self.COL_FILE_NAME: fname.lower(),
                    self.COL_FOLDER: folder.lower(),
                    **meta_cols
                }
                rows.append(row)

        if not rows:
            # Return empty with minimal columns
            return pd.DataFrame(columns=[
                self.COL_FILE_PATH, self.COL_FOLDER, self.COL_FILE_NAME
            ])

        df = pd.DataFrame(rows)
        # Ensure all string columns are lowercase
        df = df.apply(lambda col: col.astype(str).str.lower())
        self.index_df = df
        return df

    def extract_events_from_file(self, row: pd.Series) -> list:
        """
        Given one row of `self.index_df`, read that ADI file and extract all stimulation
        events based on comment markers of the form "<stim_type>_start"/"<stim_type>_stop".

        Parameters
        ----------
        row : pd.Series
            A row from `self.index_df` containing at least 'file_path' and 'file_name'.

        Returns
        -------
        list of dict
            Each dict has keys:
            ['file_path', 'file_name', 'stim_type', 'block', 'start_sample', 'stop_sample',
             'data_ch', 'stim_ch', 'fs', 'condition_0', 'condition_1', ...]
        """
        file_path = row[self.COL_FILE_PATH]
        file_name = row[self.COL_FILE_NAME]

        try:
            fread = adi.read_file(file_path)
            channel_obj = fread.channels[self.data_ch]
            fs = int(channel_obj.fs[0])
        except Exception as e:
            print(f"⚠️ Failed to read '{file_path}': {e}")
            return []

        events = []
        for block_idx, record in enumerate(channel_obj.records):
            comments = record.comments
            if not comments:
                continue

            com_texts = [c.text.lower() for c in comments]
            com_ticks = {c.text.lower(): c.tick_position for c in comments}

            for stim in self.stim_types:
                start_key = f"{stim}_start"
                stop_key  = f"{stim}_stop"
                if start_key in com_texts:
                    start_sample = com_ticks[start_key] + 1
                    if stim == 'io':
                        start_sample = max(1, start_sample - (fs // 2))
                    stop_sample = com_ticks.get(stop_key)

                    event = {
                        self.COL_FILE_PATH: file_path,
                        self.COL_FILE_NAME: file_name,
                        self.COL_STIM_TYPE: stim,
                        self.COL_BLOCK: block_idx,
                        self.COL_START: start_sample,
                        self.COL_STOP: stop_sample,
                        self.COL_DATA_CH: self.data_ch,
                        self.COL_STIM_CH: self.stim_ch,
                        self.COL_FS: fs
                    }
                    # Copy over all condition tokens
                    for col in row.index:
                        if col.startswith('condition_'):
                            event[col] = row[col]
                    events.append(event)

        return events

    def build_event_index(self) -> pd.DataFrame:
        """
        Build the complete stim event index by:
          1. Calling `scan_files()` to populate `self.index_df`.
          2. Iterating over each file row and extracting events via `extract_events_from_file()`.
          3. Combining all event dictionaries into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Stim event index with columns:
            ['file_path', 'file_name', 'stim_type', 'block', 'start_sample', 'stop_sample',
             'data_ch', 'stim_ch', 'fs', 'condition_0', 'condition_1', ...]
        """
        # Step 1: Scan for ADI files and build file index
        self.scan_files()
        all_events = []

        # Step 2: Extract events from each file
        for _, row in self.index_df.iterrows():
            events = self.extract_events_from_file(row)
            if events:
                all_events.extend(events)

        # Step 3: Create DataFrame
        if not all_events:
            # If no events, return empty with expected columns
            columns = [
                self.COL_FILE_PATH, self.COL_FILE_NAME, self.COL_STIM_TYPE, self.COL_BLOCK,
                self.COL_START, self.COL_STOP, self.COL_DATA_CH, self.COL_STIM_CH, self.COL_FS
            ]
            # Add any condition_* columns if present in index_df
            cond_cols = [c for c in self.index_df.columns if c.startswith('condition_')]
            return pd.DataFrame(columns=columns + cond_cols)

        self.event_df = pd.DataFrame(all_events)
        return self.event_df

    def save_index(self, output_csv: str):
        """
        Save the built stim event index (`self.event_df`) to a CSV file.

        Parameters
        ----------
        output_csv : str
            Path where the CSV will be written.
        """
        if self.event_df.empty:
            print("⚠️ No events to save.")
            return
        self.event_df.to_csv(output_csv, index=False)
        print(f"✅ Stim event index saved to: {output_csv}")


if __name__ == '__main__':
    # ================= USER SETTINGS =================
    main_path  = r"R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys"
    stim_types = ['io', 'rh', 'ch', 'sch']
    data_ch    = 0
    stim_ch    = 1
    sep        = '_'
    output_csv = os.path.join(main_path, 'index.csv')
    # ================================================

    indexer = StimEventIndexer(main_path, stim_types, data_ch, stim_ch, sep)
    event_index_df = indexer.build_event_index()
    indexer.save_index(output_csv)

    # Summary
    total_files  = event_index_df['file_name'].nunique() if not event_index_df.empty else 0
    total_events = len(event_index_df)
    print(f"📁 Indexed {total_files} files, found {total_events} stim events.")
