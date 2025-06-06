# -*- coding: utf-8 -*-
# =============================================================================
#                                 Imports
# =============================================================================
import adi
import numpy as np
import pandas as pd
from tqdm import tqdm
from current_clamp import Iclamp
# =============================================================================
# =============================================================================


def load_adi_data(file_path, block, start_sample, stop_sample, data_ch, stim_ch, stim_correction=1.0):
    """
    Load data and stim arrays from ADI file for a specific block and sample range.

    Parameters
    ----------
    file_path : str
    block : int
    start_sample : int
    stop_sample : int or None
    data_ch : int
    stim_ch : int
    stim_correction : float

    Returns
    -------
    raw_data : np.ndarray
    stim : np.ndarray
    fs : int
    """
    fread = adi.read_file(file_path)
    
    # get sampling frequency from data channel
    fs = int(fread.channels[data_ch].fs[0])

    # fetch data
    raw_data = fread.channels[data_ch].get_data(block + 1, start_sample=start_sample, stop_sample=stop_sample)
    stim = fread.channels[stim_ch].get_data(block + 1, start_sample=start_sample, stop_sample=stop_sample)
    stim = stim * stim_correction

    return raw_data, stim, fs


def analyze_ic(raw_data, stim, fs, stim_type, wave, prominence, interpolation_factor=1,
               post_rheo_steps=-1, max_spikes_per_step=-1):
    """
    Extract outputs from different IC protocols per cell.

    Parameters
    ----------
    raw_data : array, voltage signal
    stim : array, input current
    fs : int, samping frequency (seconds)
    stim_type : str, name of comments in labchart
    wave : bool, True to get waveform for io stims
    prominence: float, spike prominence for detection
    interpolation_factor: int, interpolation factor, default = 1.
    post_rheo_steps: int, max number of IO steps to collect spikes post rheobase
    Default is -1 = collect from all steps
    max_spikes_per_step : int, number of max spikes to collect per step. 
    Default is -1 = collect all spikes.

    Returns
    -------
    output : dict, extracted data

    """
    
    # init current clamp object
    ic = Iclamp(fs, dist=1, prominence=prominence)
    
    if stim_type =='io':
        signal, i_amp, dur = ic.parse_io(raw_data, stim)
        locs = ic.spike_locs(signal)
    
    if (stim_type =='io') & (wave==False):
        spike_freq = np.array(ic.count_spikes(locs))/ (dur/fs)
        input_res = ic.get_input_resistance(signal, i_amp, ic.count_spikes(locs))
        rmp = ic.get_rmp(raw_data)
        output = {'spike_frequency':spike_freq, 'input_resistance': input_res,
                  'amp':i_amp, 'rmp':[rmp]*len(spike_freq)}
        
    if (stim_type == 'io') & (wave==True):
        waveform, amp, times = ic.select_waveforms(signal, i_amp, locs,
                                                   interpolation_factor=interpolation_factor,
                                                   post_rheo_steps=post_rheo_steps, 
                                                   max_spikes_per_step=max_spikes_per_step)

        if waveform:
            output = {'mV':waveform, 'amp':amp, 'time': times}
        else:
            output ={}
 
    if stim_type == 'rh':
        irheo = ic.get_rheobase(raw_data, stim)
        output = {'rheobase':[irheo]}
        
    if stim_type == 'sch':
        spike_count, freqs = ic.get_short_chirp(raw_data, stim, window=0.25)
        output = {'spike_count':spike_count, 'freq':freqs}

    return output


if __name__ == '__main__':
    
    # settings
    index_csv_path = r"R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\stim_event_index_verified.csv"
    index_df = pd.read_csv(index_csv_path)
    stim_correction = 1000
    
    # get data and stim
    row = index_df.loc[0]
    raw_data, stim, fs = load_adi_data(row.file_path, row.block, row.start_sample,
                                       row.stop_sample, row.data_ch, row.stim_ch, stim_correction)
    
    # analyze
    analyze_ic(raw_data, stim, fs, row.stim_type, False, row.threshold, interpolation_factor=1,
                   post_rheo_steps=-1, max_spikes_per_step=-1)
    
    
    # # results = run_batch_analysis(index_file, wave=False,
    # #                               interpolation_factor=2,
    # #                               post_rheo_steps=3,
    # #                               max_spikes_per_step=2)

    # # for stim_type, df in results.items():
    # #     out_path = index_file.replace('.csv', f'_{stim_type}_features.csv')
    # #     df.to_csv(out_path, index=False)
    # #     print(f"✅ Saved: {out_path}")



