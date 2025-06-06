# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
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
##### ------------------------------------------------------------------- #####


def load_adi_data(file_path, block, start_sample, stop_sample, data_ch, stim_ch, stim_correction=1.0):
    """
    Load voltage and stimulus arrays from ADI file for one block.
    """
    fread = adi.read_file(file_path)
    ch_data = fread.channels[data_ch]
    ch_stim = fread.channels[stim_ch]

    fs = int(ch_data.fs[0])
    start = max(1, int(start_sample))
    stop = None if pd.isna(stop_sample) else int(stop_sample)

    raw_data = ch_data.get_data(block + 1, start_sample=start, stop_sample=stop)
    stim = ch_stim.get_data(block + 1, start_sample=start, stop_sample=stop) * stim_correction

    return raw_data, stim, fs


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
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


def analyze_ic(raw_data, stim, fs, stim_type, wave, prominence, interpolation_factor=1,
               post_rheo_steps=-1, max_spikes_per_step=-1, show_io=False):
    ic = Iclamp(fs, dist=1, prominence=prominence, show_io=show_io)

    if stim_type == 'io':
        signal, i_amp, dur = ic.parse_io(raw_data, stim)
        locs = ic.spike_locs(signal)
        if wave:
            waveform, amp, times = ic.select_waveforms(
                signal, i_amp, locs,
                interpolation_factor=interpolation_factor,
                post_rheo_steps=post_rheo_steps,
                max_spikes_per_step=max_spikes_per_step
            )
            return {'mV': waveform, 'amp': amp, 'time': times} if waveform else {}
        spike_freq = np.array(ic.count_spikes(locs)) / (dur / fs)
        input_res = ic.get_input_resistance(signal, i_amp, ic.count_spikes(locs))
        rmp = ic.get_rmp(raw_data)
        # time_constant = ic.get_time_constant(signal, i_amp)
        return {
            'spike_frequency': spike_freq,
            'input_resistance': input_res,
            'amp': i_amp,
            # 'time_constant': time_constant,
            'rmp': [rmp] * len(spike_freq)
        }

    elif stim_type == 'rh':
        return {'rheobase': [ic.get_rheobase(raw_data, stim)]}

    elif stim_type == 'sch':
        spike_count, freqs = ic.get_spike_transfer(raw_data, stim)
        return {'spike_count': spike_count, 'freq': freqs}

    elif stim_type == 'ch':
        impedance, power, freq = ic.get_chirp(raw_data, stim)
        return {'impedance': impedance, 'peak_power': power, 'freq': freq}

    return {}


class BatchProcess:
    """
    Batch processing class for current clamp datasets.
    """

    def __init__(self, index_df, njobs=1, show_io=False, stim_correction=1000, prominence=30):
        self.index = index_df.copy()
        self.max_jobs = max(1, multiprocessing.cpu_count() - 2)
        self.show_io = show_io
        self.stim_correction = stim_correction
        self.prominence = prominence
        self.njobs = min(max(njobs, 1), self.max_jobs)

    def extract_data(self, idx, row):
        raw_data, stim, fs = load_adi_data(
            file_path=row.file_path,
            block=row.block,
            start_sample=row.start_sample,
            stop_sample=row.stop_sample,
            data_ch=row.data_ch,
            stim_ch=row.stim_ch,
            stim_correction=self.stim_correction
        )

        if not np.any(raw_data):
            raise Exception(f'--> Could not read data from: {row.file_path}')

        output = analyze_ic(
            raw_data, stim, fs,
            stim_type=row.stim_type,
            wave=self.wave,
            prominence=self.prominence,
            interpolation_factor=self.interpolation_factor,
            post_rheo_steps=self.post_rheo_steps,
            max_spikes_per_step=self.max_spikes_per_step,
            show_io=self.show_io
        )

        if output:
            output['id'] = np.repeat(idx, len(output[next(iter(output))]))
            return pd.DataFrame(data=output)

        print(f'--> No output extracted from: {row.file_path}')
        return pd.DataFrame()

    def all_cells(self, stim_type, wave=False, interpolation_factor=1,
                  post_rheo_steps=-1, max_spikes_per_step=-1):
        self.wave = wave
        self.interpolation_factor = interpolation_factor
        self.post_rheo_steps = post_rheo_steps
        self.max_spikes_per_step = max_spikes_per_step

        df_stim = self.index[self.index.stim_type == stim_type].copy()

        if self.njobs == 1:
            df_list = [
                self.extract_data(idx, row)
                for idx, row in tqdm(df_stim.iterrows(), total=len(df_stim), desc='Progress:')
            ]
        else:
            with tqdm_joblib(tqdm(desc='Progress:', total=len(df_stim))):
                df_list = Parallel(n_jobs=self.njobs, backend='loky')(
                    delayed(self.extract_data)(idx, row) for idx, row in df_stim.iterrows()
                )

        df = pd.concat(df_list, axis=0)
        return df_stim.join(df.set_index('id')).reset_index(drop=True)

    def run_analysis_and_save(self, stim_type, output_dir, wave=False,
                              interpolation_factor=1, post_rheo_steps=-1, max_spikes_per_step=-1):
        df = self.all_cells(
            stim_type=stim_type,
            wave=wave,
            interpolation_factor=interpolation_factor,
            post_rheo_steps=post_rheo_steps,
            max_spikes_per_step=max_spikes_per_step
        )

        suffix = 'wave' if wave else 'basic'
        name = f"{stim_type}_{suffix}" if stim_type == 'io' else stim_type
        filename = os.path.join(output_dir, f"{name}_features.csv")

        df.to_csv(filename, index=False)
        print(f"✅ Saved {stim_type.upper()} results to: {filename}")
        return df


# =============================================================================
#                                 MAIN BLOCK
# =============================================================================
if __name__ == '__main__':
    main_path = r"R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys"
    index_path = os.path.join(main_path, "index.csv")
    index_df = pd.read_csv(index_path)

    processor = BatchProcess(index_df, njobs=1, show_io=False, prominence=30)

    # Example: run and save rheobase results
    rh_df = processor.run_analysis_and_save(
        stim_type='io',
        output_dir=main_path,
        wave=True,
        max_spikes_per_step=3,
        post_rheo_steps=3,
        interpolation_factor=10
        
    )
