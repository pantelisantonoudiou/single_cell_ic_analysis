# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os                      # For file and directory operations
import adi                     # For reading .adicht ADInstruments files
import pandas as pd            # For data handling and storage
##### ------------------------------------------------------------------- #####


class StimEventIndexer:
    """
    Indexes stimulation events in `.adicht` files organized within subfolders.

    Each subfolder under `main_path` is treated as an experimental condition
    (e.g., "pv_gfp" → condition_0='pv', condition_1='gfp'). Each `.adicht` file
    within those folders is scanned for comment markers like "io_start" / "io_stop",
    and those are recorded along with metadata.

    Parameters
    ----------
    main_path : str
        Root directory containing folders of `.adicht` files.
    stim_types : list of str
        List of expected stim type prefixes (e.g., ['io', 'rh']).
    data_ch : int
        Index of the voltage channel in the file (typically Vm).
    stim_ch : int
        Index of the stimulus/current channel.
    sep : str
        Separator used to parse folder names into condition columns.

    Returns
    -------
    pd.DataFrame
        One row per stimulation event, including metadata like file name,
        stim type, sample range, condition labels, and channel/sample rate info.
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
        self.main_path = os.path.normpath(main_path)  # Normalize path separators
        self.stim_types = stim_types                  # Stimulation protocols of interest
        self.data_ch = data_ch                        # Voltage channel index
        self.stim_ch = stim_ch                        # Stimulus/current channel index
        self.sep = sep                                # Condition delimiter in folder names

    def scan_files(self) -> pd.DataFrame:
        """
        Scans all immediate subdirectories of `main_path` for `.adicht` files.
        Each folder name is parsed into experimental condition columns.

        Returns
        -------
        pd.DataFrame
            File index with columns:
            ['file_path', 'file_name', 'folder_name', 'condition_0', 'condition_1', ...]
        """
        rows = []
        for folder in os.listdir(self.main_path):
            folder_path = os.path.join(self.main_path, folder)
            if not os.path.isdir(folder_path):
                continue  # Skip files at root level

            # Split folder name into condition tokens
            tokens = folder.lower().split(self.sep)
            meta_cols = {f'condition_{i}': tokens[i] for i in range(len(tokens))}

            # Collect .adicht files within this folder
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
            return pd.DataFrame(columns=[
                self.COL_FILE_PATH, self.COL_FOLDER, self.COL_FILE_NAME
            ])

        df = pd.DataFrame(rows)
        df = df.apply(lambda col: col.astype(str).str.lower())  # Normalize to lowercase
        return df

    def extract_events_from_file(self, row: pd.Series) -> list:
        """
        Reads one ADI file and extracts all stim events based on start/stop comments.

        Parameters
        ----------
        row : pd.Series
            One row from the index dataframe produced by `scan_files()`.

        Returns
        -------
        list of dict
            One dictionary per stimulation event, including sample ranges,
            stim type, block index, and condition metadata.
        """
        file_path = row[self.COL_FILE_PATH]
        file_name = row[self.COL_FILE_NAME]

        try:
            fread = adi.read_file(file_path)
            channel_obj = fread.channels[self.data_ch]
            fs = int(channel_obj.fs[0])  # Sampling frequency
        except Exception as e:
            print(f"⚠️ Failed to read '{file_path}': {e}")
            return []

        events = []
        for block_idx, record in enumerate(channel_obj.records):
            comments = record.comments
            if not comments:
                continue  # No markers

            # Build comment lookup
            com_texts = [c.text.lower() for c in comments]
            com_ticks = {c.text.lower(): c.tick_position for c in comments}

            for stim in self.stim_types:
                start_key = f"{stim}_start"
                stop_key  = f"{stim}_stop"
                if start_key in com_texts:
                    start_sample = com_ticks[start_key] + 1
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
                    # Add conditions from folder metadata
                    for col in row.index:
                        if col.startswith('condition_'):
                            event[col] = row[col]
                    events.append(event)

        return events

    def build_event_index(self) -> pd.DataFrame:
        """
        Scans all `.adicht` files and builds a unified stim event index.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all stim events across files/folders.
        """
        index_df = self.scan_files()
        all_events = []

        for _, row in index_df.iterrows():
            events = self.extract_events_from_file(row)
            if events:
                all_events.extend(events)

        if not all_events:
            base_cols = [
                self.COL_FILE_PATH, self.COL_FILE_NAME, self.COL_STIM_TYPE, self.COL_BLOCK,
                self.COL_START, self.COL_STOP, self.COL_DATA_CH, self.COL_STIM_CH, self.COL_FS
            ]
            cond_cols = [c for c in index_df.columns if c.startswith('condition_')]
            return pd.DataFrame(columns=base_cols + cond_cols)

        return pd.DataFrame(all_events)


if __name__ == '__main__':
    # ================= USER SETTINGS =================
    main_path  = r"R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys"   # Root path for data
    stim_types = ['io', 'rh', 'ch', 'sch']                                 # Stimulation protocols
    data_ch    = 0                                                         # Voltage channel index
    stim_ch    = 1                                                         # Stimulus/current channel index
    sep        = '_'                                                       # Folder name delimiter
    output_csv = os.path.join(main_path, 'index.csv')                      # Save location
    # ================================================

    indexer = StimEventIndexer(main_path, stim_types, data_ch, stim_ch, sep)
    event_df = indexer.build_event_index()
    event_df.to_csv(output_csv, index=False)

    # Summary
    total_files  = event_df['file_name'].nunique() if not event_df.empty else 0
    total_events = len(event_df)
    print(f"📁 Indexed {total_files} files, found {total_events} stim events.")
