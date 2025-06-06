# -*- coding: utf-8 -*-
# =============================================================================
#                                 Imports
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import adi
from current_clamp import Iclamp
##### ------------------------------------------------------------------- #####


def load_adi_data(file_path, block, start_sample, stop_sample, data_ch, stim_ch, stim_correction=1.0):
    """
    Load voltage (raw_data) and stimulus (stim) arrays from an ADI file
    for a specific block and sample range.

    Parameters
    ----------
    file_path : str
        Absolute path to the .adicht file.
    block : int
        Zero-based block index in the ADI file.
    start_sample : int
        Sample index where the stimulation begins.
    stop_sample : int or None
        Sample index where the stimulation ends; if None, read until block end.
    data_ch : int
        Index of the voltage channel in the ADI file.
    stim_ch : int
        Index of the stimulus/current channel in the ADI file.
    stim_correction : float
        Multiplicative factor to scale the raw stim values (e.g., to convert to pA).

    Returns
    -------
    raw_data : np.ndarray
        Voltage trace for that block and sample range (in mV).
    stim : np.ndarray
        Stimulus trace (scaled by stim_correction).
    fs : int
        Sampling frequency of the voltage channel (Hz).
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


def analyze_ic(raw_data, stim, fs, stim_type, wave, prominence,
               interpolation_factor=1, post_rheo_steps=-1, max_spikes_per_step=-1):
    """
    Extract outputs from different IC protocols for a single stimulation segment.

    Parameters
    ----------
    raw_data : np.ndarray
        Voltage trace (mV).
    stim : np.ndarray
        Stimulus trace (pA).
    fs : int
        Sampling frequency (Hz).
    stim_type : str
        One of ['io', 'rh', 'sch', 'ch'] indicating the protocol.
    wave : bool
        If True, extract full waveforms for IO steps.
    prominence : float
        Spike detection prominence threshold (mV).
    interpolation_factor : int
        Upsampling factor for waveform interpolation.
    post_rheo_steps : int
        Number of IO steps to collect post-rheobase spikes. -1 = all.
    max_spikes_per_step : int
        Maximum spikes per IO step. -1 = all.

    Returns
    -------
    output : dict
        Depending on stim_type:
            - 'io': keys 'spike_frequency', 'input_resistance', 'amp', 'rmp'
              or, if wave=True, 'mV', 'amp', 'time'.
            - 'rh': key 'rheobase'
            - 'sch': keys 'spike_count', 'freq'
            - 'ch': keys 'impedance', 'peak_power', 'freq'
    """
    ic = Iclamp(fs, dist=1, prominence=prominence)

    output = {}
    if stim_type == 'io':
        signal, i_amp, dur = ic.parse_io(raw_data, stim)
        locs = ic.spike_locs(signal)

        if not wave:
            spike_counts = ic.count_spikes(locs)
            spike_freq = np.array(spike_counts) / (dur / fs)
            input_res = ic.get_input_resistance(signal, i_amp, spike_counts)
            rmp = ic.get_rmp(raw_data)
            output = {
                'spike_frequency': spike_freq,
                'input_resistance': input_res,
                'amp': i_amp,
                'rmp': [rmp] * len(spike_freq)
            }
        else:
            waveform, amp, times = ic.select_waveforms(
                signal, i_amp, locs,
                interpolation_factor=interpolation_factor,
                post_rheo_steps=post_rheo_steps,
                max_spikes_per_step=max_spikes_per_step
            )
            if waveform:
                output = {'mV': waveform, 'amp': amp, 'time': times}

    elif stim_type == 'rh':
        irheo = ic.get_rheobase(raw_data, stim)
        output = {'rheobase': [irheo]}

    elif stim_type == 'sch':
        spike_count, freqs = ic.get_short_chirp(raw_data, stim, window=0.25)
        output = {'spike_count': spike_count, 'freq': freqs}

    elif stim_type == 'ch':
        impedance, peak_power, freqs = ic.get_chirp(raw_data, stim)
        output = {'impedance': impedance, 'peak_power': peak_power, 'freq': freqs}

    return output


def process_first_file(index_csv_path, data_ch=0, stim_ch=1, stim_correction=1e3, prominence=30):
    """
    Load the first row of the stim event index, retrieve its data segment, analyze it,
    and plot results appropriate for its stim_type.

    Parameters
    ----------
    index_csv_path : str
        Path to the CSV file containing the stim event index.
    data_ch : int
        ADI channel index for voltage (Vm) data.
    stim_ch : int
        ADI channel index for stimulus data.
    stim_correction : float
        Scaling factor to convert raw stim units to pA.
    prominence : float
        Default spike prominence threshold (mV).

    Returns
    -------
    None
    """
    # Load index
    index_df = pd.read_csv(index_csv_path)
    if index_df.empty:
        print("Index is empty; nothing to process.")
        return

    # Use the first event row
    row = index_df.iloc[0]
    file_path = row['file_path']
    block = int(row['block'])
    start_sample = int(row['start_sample'])
    stop_sample = row['stop_sample'] if not pd.isna(row['stop_sample']) else None
    stim_type = row['stim_type']

    # Load raw_data, stim, fs
    raw_data, stim, fs = load_adi_data(
        file_path, block, start_sample, stop_sample,
        data_ch, stim_ch, stim_correction
    )

    # Analyze
    result = analyze_ic(
        raw_data, stim, fs,
        stim_type=stim_type,
        wave=False,
        prominence=prominence
    )
    
    return result

if __name__ == '__main__':
    # === Configuration ===
    index_csv = r"R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\index.csv"
    data_ch = 0
    stim_ch = 1
    stim_correction = 1000  # scale to pA
    prominence = 30         # default spike detection prominence
    # ========================

    result = process_first_file(
        index_csv_path=index_csv,
        data_ch=data_ch,
        stim_ch=stim_ch,
        stim_correction=stim_correction,
        prominence=prominence
    )
