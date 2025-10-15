# -*- coding: utf-8 -*-
"""
batch_process.py

Batch‐processing of current‐clamp experiments using Iclamp.
***Voltage data need to be converted to mV and Stim to pA***

Features
--------
- Loads raw_data & stim from ADI via load_adi_data()
- Runs each protocol (‘io’, ‘rh’, ‘sch’, ‘ch’)
- For IO, does both basic (spike_freq etc.) and wave (waveforms)
- Saves results per‐protocol to CSV
- Generates validation plots (one per cell) under analyzed/<stim>_plots/
"""

# =============================== IMPORTS =============================== #
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import joblib
from joblib import Parallel, delayed
import contextlib
from current_clamp import Iclamp
import adi
# ======================================================================= #


def load_adi_data(file_path, block, start_sample, stop_sample, data_ch, stim_ch, data_correction=1.0, stim_correction=1.0):
    """
    Load voltage and stimulus data from an ADI file block.

    Parameters
    ----------
    file_path : str
        Full path to the .adicht file.
    block : int
        Block number (zero-based).
    start_sample : int
        Start sample index.
    stop_sample : int or None
        Stop sample index. If None, reads until block end.
    data_ch : int
        Index of voltage (membrane potential) channel.
    stim_ch : int
        Index of stimulus (current injection) channel.
    data_correction : float, optional
            Scaling factor for voltage data, by default 1.0.
    stim_correction : float, optional
        Scaling factor for stimulus data, by default 1.0.

    Returns
    -------
    raw_data : np.ndarray
        Voltage signal.
    stim : np.ndarray
        Stimulus trace.
    fs : int
        Sampling frequency.
    """
    
    fread = adi.read_file(file_path)
    ch_data = fread.channels[data_ch]
    ch_stim = fread.channels[stim_ch]
    fs = int(ch_data.fs[0])

    start = max(1, int(start_sample))
    stop = None if pd.isna(stop_sample) else int(stop_sample)

    raw_data = ch_data.get_data(block + 1, start_sample=start, stop_sample=stop) * data_correction
    stim = ch_stim.get_data(block + 1, start_sample=start, stop_sample=stop) * stim_correction

    return raw_data, stim, fs


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Patch joblib to report into tqdm progress bar.
    """
    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            tqdm_object.update(self.n_completed_tasks - tqdm_object.n)

    original = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress
    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original
        tqdm_object.close()


class BatchProcess:
    """
    Batch‐analysis of current clamp data via Iclamp.analyze + save_validation_plot.
    """

    def __init__(self, main_path, index_df, data_correction=1.0,
                 njobs=1, stim_correction=1000, prominence=30):
        """
        main_path : str
            Parent folder for all .adicht data and where `index.csv` lives.
        index_df : pd.DataFrame
            Must contain ['file_path','file_name','stim_type','block',
                           'start_sample','stop_sample','data_ch','stim_ch'].
        njobs : int
            Parallel workers.
        data_correction : float
            Scale ADI data channel.
        stim_correction : float
            Scale ADI stim channel.
        prominence : float
            Spike detection threshold.
        """
        self.main_path       = main_path
        self.index           = index_df.copy().reset_index(drop=True) # get unique IDs
        self.data_correction = data_correction
        self.stim_correction = stim_correction
        self.prominence      = prominence

        maxj = max(1, multiprocessing.cpu_count() - 2)
        self.njobs = min(max(1, njobs), maxj)

    def extract_one(self, idx, row, wave, interp, post_steps, max_spikes):
        """
        Load, analyze, and save plot for **one** stim‐event row.
        """
        
        # load data and stim
        raw, stim, fs = load_adi_data(
            file_path=row.file_path,
            block=row.block,
            start_sample=row.start_sample,
            stop_sample=row.stop_sample,
            data_ch=row.data_ch,
            stim_ch=row.stim_ch,
            data_correction=self.data_correction,
            stim_correction=self.stim_correction
        )

        if raw.size == 0:
            raise RuntimeError(f"Empty data: {row.file_path}")
        
        # analyze
        ic = Iclamp(fs, prominence=self.prominence)
        metrics = ic.analyze(
            raw, stim,
            row.stim_type,
            wave=wave,
            interpolation_factor=interp,
            post_rheo_steps=post_steps,
            max_spikes_per_step=max_spikes
        )

        # Attach ID for joining later
        n = len(list(metrics.values())[0]) if metrics else 0
        if n == 0:
            return pd.DataFrame()  # nothing extracted

        metrics['id'] = np.repeat(idx, n)
        dfm = pd.DataFrame(metrics)

        # save validation figure
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.splitext(row.file_name)[0]
        plot_path = os.path.join(plot_dir, f"{fname}_blk{row.block}.png")
        ic.save_validation_plot(raw, stim, row.stim_type, plot_path,
                                wave=wave,
                                post_rheo_steps=post_steps,
                                max_spikes_per_step=max_spikes)

        return dfm

    def all_cells(self, stim_type, wave=False,
              interpolation_factor=1, post_rheo_steps=-1,
              max_spikes_per_step=-1):
        """
        Run analysis across all events for a given stimulation type.
    
        Parameters
        ----------
        stim_type : str
            Type of stimulation ('io', 'rh', 'sch', 'ch').
        wave : bool
            Whether to extract waveforms (IO only).
        interpolation_factor : int
            Interpolation factor for waveforms.
        post_rheo_steps : int
            Number of IO steps to extract post-rheobase.
        max_spikes_per_step : int
            Maximum number of spikes to extract per IO step.
    
        Returns
        -------
        pd.DataFrame
            Index joined with extracted features.
        """
        subdf = self.index[self.index.stim_type == stim_type]
    
        # Serial execution
        if self.njobs == 1:
            results = []
            for idx, row in tqdm(subdf.iterrows(), total=len(subdf), desc=f"{stim_type}{'_wave' if wave else ''}"):
                df = self.extract_one(
                    idx, row, wave,
                    interpolation_factor,
                    post_rheo_steps,
                    max_spikes_per_step
                )
                results.append(df)
    
        # Parallel execution
        else:
            jobs = [
                delayed(self.extract_one)(
                    idx, row, wave,
                    interpolation_factor,
                    post_rheo_steps,
                    max_spikes_per_step
                )
                for idx, row in subdf.iterrows()
            ]
            with tqdm_joblib(tqdm(total=len(jobs), desc=f"{stim_type}{'_wave' if wave else ''}")):
                parallel_results = Parallel(n_jobs=self.njobs)(jobs)
            results = [res[0] for res in parallel_results]
        
        feat = pd.concat(results, ignore_index=True)
        result =  subdf.join(feat.set_index('id'), how='left').reset_index()
        return  result

    def run_all(self, output_dir, stim_types):
        """
        Run and save feature CSVs & plots for each protocol.
        For IO, does both basic and waveform modes.
        """
        for st in stim_types:
            if st == 'io':
                # basic
                self.output_dir = os.path.join(output_dir, 'io_basic')
                os.makedirs(self.output_dir, exist_ok=True)
                df_basic = self.all_cells(st, wave=False)
                df_basic.to_csv(os.path.join(self.output_dir, 'features.csv'), index=False)

                # waveform
                self.output_dir = os.path.join(output_dir, 'io_wave')
                os.makedirs(self.output_dir, exist_ok=True)
                df_wave = self.all_cells(
                    st, wave=True,
                    interpolation_factor=10,
                    post_rheo_steps=3,
                    max_spikes_per_step=3
                )
                df_wave.to_csv(os.path.join(self.output_dir, 'features.csv'), index=False)

            else:
                self.output_dir = os.path.join(output_dir, st)
                os.makedirs(self.output_dir, exist_ok=True)
                df = self.all_cells(st)
                df.to_csv(os.path.join(self.output_dir, 'features.csv'), index=False)

        print("✅ All protocols processed.")


if __name__ == '__main__':
    # === USER SETTINGS ===
    main_path   = r"R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\full_dataset"
    index_csv   = os.path.join(main_path, "index.csv")
    analyzed_dir = os.path.join(main_path, 'analyzed')
    stim_types  = ['io', 'rh', 'ch', 'sch'] #'io', 'rh', 'ch', 'sch'
    njobs       = 1
    # =====================

    idx_df = pd.read_csv(index_csv)
    processor = BatchProcess(
        main_path, idx_df,
        njobs=njobs,
        stim_correction=1000,
        prominence=25
    )
    processor.run_all(analyzed_dir, stim_types)
