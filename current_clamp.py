 
# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks, stft
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 16})
##### ------------------------------------------------------------------- #####

class Iclamp:
    """
    Detect and calculate spike properties for different current clamp stimulation protocols.
    """
    
    def __init__(self, fs, dist=1, spike_select=[2, 3], prominence=40, wlen=.1):
        """
        Parameters
        ----------
        fs : int, sampling frequency (per second)
        dist : float, find peaks distance (ms)
        spike_select : float, type to get before and after spike location (ms)
        prominence : float, spike threshold from neighbour (mV)
        wlen : percentage of fs samples for prominence detection 
        """
        
        self.fs = fs
        self.dist = int(dist*fs/1000)    
        self.spike_select = np.array(np.array(spike_select)*fs/1000, dtype=int)
        self.prominence = prominence
        self.wlen = int(fs*wlen)
            
    def get_stim_times_io(self, stim, threshold=2):
        """
        Get stimulus times.

        Parameters
        ----------
        stim : array, input current
        threshold : float, for spike detection based on stim gradient. 

        Returns
        -------
        start : array, with stim start times
        stop : array, with stim stop times
        dur : int, median stim duration

        """
        
        stim_d = np.gradient(np.abs(stim))
        start, _ = find_peaks(stim_d, height=threshold, distance=self.dist)
        stop, _= find_peaks(-stim_d, height=threshold, distance=self.dist)
        dur = int(np.median(stop-start))
        
        return start, stop, dur
    
    
    def parse_io(self, data, stim):
        """
        Organize IO signal and stimulus into numpy arrays.

        Parameters
        ----------
        data : array, voltage signal
        stim : array, input current

        Returns
        -------
        signal, 2D array (signal for each I step, time)
        i_amp : array, with current amplitude for each I step.
        dur: int, duration of each io step in samples.

        """
        
        # get stim times
        start, stop, dur = self.get_stim_times_io(stim, threshold=2)
        
        # get baseline
        self.base = np.median(data[start[0]-int(self.fs/2):start[0]])
        
        signal = []
        i_amp = []
        for i in start:
            signal.append(data[i : i+dur])
            i_amp.append(stim[i : i+dur])
        
        i_amp = np.mean(np.array(i_amp), axis=1, dtype=int)
        i_amp = np.round(i_amp,-1)

        return np.array(signal), i_amp, dur

    def parse_io_for_raw(self, data, stim, peri_stim_time=50):
        """
        Organize IO signal and stimulus into numpy arrays.

        Parameters
        ----------
        data : array, voltage signal
        stim : array, input current
        peri_stim_time: int, number of samples around stim time

        Returns
        -------
        signal, 2D array (signal for each I step, time)
        i_amp : 2D array (stim for each I step, time)
        mean_amp : array, with current amplitude for each I step.


        """
        
        # get stim times
        start, stop, dur = self.get_stim_times_io(stim, threshold=2)
        
        # get baseline
        self.base = np.median(data[start[0]-int(self.fs/2):start[0]])
        
        signal = []
        i_amp = []
        mean_amp = []
        for i in start:
            signal.append(data[i-peri_stim_time : i+dur+peri_stim_time])
            i_amp.append(stim[i-peri_stim_time : i+dur+peri_stim_time])
            mean_amp.append(stim[i : i+dur])
        
        mean_amp = np.mean(np.array(mean_amp), axis=1, dtype=int)
        mean_amp = np.round(mean_amp,-1)

        return np.array(signal), np.array(i_amp), mean_amp
    
    
    def spike_locs(self, signal):
        """
        Detect spike location for each io step.

        Parameters
        ----------
        signal : 2D array (signal for each I step, time)

        Returns
        -------
        locs : list, with arrays of spike times for row of signal

        """
        locs = []
        for s in signal:
          spike_locs, _ = find_peaks(s[self.spike_select[0]:-self.spike_select[1]],
                                     prominence=self.prominence, distance=self.dist,
                                     wlen=self.wlen)

          locs.append(spike_locs + self.spike_select[0])
        return locs
    
    
    def count_spikes(self, locs):
        """
        Count number of spikes for each io step.

        Parameters
        ----------
        locs : list, with arrays of spike times for row of signal

        Returns
        -------
        spike_count : list, number of spikes for each io step

        """

        spike_count = [len(x) for x in locs]
        
        return spike_count    
    
    
    def get_input_resistance(self, signal, i_amp, spike_count, i_to_v=1e9):
        """
        Get input resistance for each negative I step.

        Parameters
        ----------
        signal : 2D array (signal for each I step, time)
        i_amp : array, with current amplitude for each I step.
        spike_count : list, number of spikes for each io step
        i_to_v : float, convertion factor

        Returns
        -------
        input_res : list, input resistance

        """
        input_res = []
        for s, a, sc in zip(signal, i_amp, spike_count):
            if (a < 21) & (sc == 0):
                ir_value = (np.median(s)-self.base)/ (a/i_to_v)
            else:
                ir_value = None
            input_res.append(ir_value)
            
        return input_res
    
    
    def get_mean_waveform_per_io(self, signal, i_amp, locs):
        """
        Get mean spike waveform for each io step.

        Parameters
        ----------
        signal : 2D array (signal for each I step, time)
        i_amp : array, with current amplitude for each I step.
        locs : list, with arrays of spike times for row of signal

        Returns
        -------
        waveform : list, mean spike waveform for each i step
        amp : list, current waveform
        times: list, time series for each spike

        """
        
        waveform = []
        amp = []
        times = []
        time = list(range(-self.spike_select[0], self.spike_select[1]))
        
        for s, a, l in zip(signal, i_amp, locs):
            if len(l) > 0:
                spikes = [s[loc-self.spike_select[0]:loc+self.spike_select[1]] for loc in l]
                spikes = np.mean(np.array(spikes), axis=0)
                waveform.extend(spikes)
                amp.extend([a] * spikes.shape[0])
                times.extend(time)

        return waveform, amp, times
    
    def select_waveforms(self, signal, i_amp, locs, interpolation_factor=1,
                         post_rheo_steps=-1, max_spikes_per_step=-1,):
        """
        Get waveforms from the first n steps post rheobase.

        Parameters
        ----------
        signal : 2D array (signal for each I step, time).
        i_amp : array, with current amplitude for each I step.
        locs : list, with arrays of spike times for row of signal.
        interpolation_factor: int, interpolation factor, default = 1.
        post_rheo_steps: int, max number of IO steps to collect spikes post rheobase
        Default is -1 = collect from all steps
        max_spikes_per_step : int, number of max spikes to collect per step. 
            Default is -1 = collect all spikes.

        Returns
        -------
        waveform : list, mean spike waveform for each i step
        amp : list, current waveform
        times: list, time series for each spike

        """
        
        # create templates
        waveform = []
        amp = []
        times = []
        time = list(range(-self.spike_select[0], self.spike_select[1]))
        
        # find rheobase
        cntr = 0
        idx = []
        for a, loc in zip(i_amp, locs):
            if cntr == post_rheo_steps:
                break
            if len(loc) > 0:
                idx.extend(np.where(i_amp == a)[0])
                cntr+=1
        idx = np.array(idx)
        
        if len(idx) == 0:
            return waveform, amp, times
        
        # collect spikes
        locs = np.array(locs, dtype=object)
        for s,a,l in zip(signal[idx], i_amp[idx], locs[idx]):
            for i, loc in enumerate(l):
 
                if i == max_spikes_per_step: # collect max spikes per train
                    break
                    
                # get spike waveform and time
                y = s[loc-self.spike_select[0]:loc+self.spike_select[1]]
                t = time
                
                # interpolate spike waveform
                if interpolation_factor > 1:
                    step = (t[1]-t[0])/interpolation_factor
                    t2 = np.arange(t[0],t[-1]-step, step)
                    f = interpolate.interp1d(t, y, kind='cubic')
                    y = f(t2)
                    t = t2

                waveform.extend(y)
                amp.extend([a] * len(t))
                times.extend(t)
    
        return waveform, amp, times
    
    
    def get_rmp(self, data, bins=np.arange(-100, 0, .1)):
        """
        Get resting membrane potential from io step.
        
        Parameters
        ----------
        data : array, voltage signal
        bins : array, for hist edges. The default is np.arange(-100,0,.1).

        Returns
        -------
        rmp: float, resting membrane potential.

        """
        bin_counts, edges = np.histogram(data, bins=bins)
        rmp = edges[np.argmax(bin_counts)]
        
        return round(rmp, 3)


    def get_rheobase(self, data, stim):
        """
        Get rheobase current from the first detected spike.

        Parameters
        ----------
        data : array, voltage signal
        stim : array, input current
        
        Returns
        -------
        rheo : float, rheobase current in pA

        """

        locs, _ = find_peaks(data, prominence=self.prominence, distance=self.dist)
        if locs.size == 0:
            out = None
        else:
            out = np.mean(stim[locs[0]-int(self.fs*0.001): locs[0]+int(self.fs*0.001)])
            
        return out

    
    def get_chirp(self, data, stim, stim_to_amp=1e12, data_to_v=1e3, freq_range=(2,59)):
        """
        Calculate the impedance across a frequency range for a given voltage signal and input stimulus.
        
        Parameters
        ----------
        data : array, Voltage signal (one trial), usually representing neural activity or membrane voltage.
        stim : array, Input current signal (one trial), used to drive the system.
        stim_to_amp, float, conversion of stim to Amps (assumes input data are in pA)
        data_to_v, covnersion of membrane voltage values to Volts (assumes input data are in mV)
        
        Returns
        -------
        impedance : array, Impedance values for the specified frequency range. (MOhm)
        peak_power: array, Peak power per bin(V^2/Hz)
        freq : array, Frequencies corresponding to the impedance values within the specified range.
        
        Notes
        -----
        1. The function uses a window size equal to the sampling frequency for STFT, with a 50% overlap.
        2. The impedance is calculated for frequencies between 2 Hz and 59 Hz by default.
        """
        
        # get power for each freq 1 Hz bin
        f, t, i_power = stft(stim/stim_to_amp, fs=self.fs, nperseg=self.fs, noverlap=(int(self.fs/2)))
        f, t, v_power = stft(data/data_to_v, fs=self.fs, nperseg=self.fs, noverlap=(int(self.fs/2)))

        # get psd
        i_psd = np.mean(np.abs(i_power)**2, axis=1)
        v_psd = np.mean(np.abs(v_power)**2, axis=1)
              
        # trim to range
        lower_idx = np.where(f==freq_range[0])[0][0]
        upper_idx = np.where(f==freq_range[1])[0][0]+1
        v_psd_trimmed = v_psd[lower_idx:upper_idx]
        i_psd_trimmed = i_psd[lower_idx:upper_idx]
        freq = f[lower_idx:upper_idx]
        
        # get impedance
        impedance = np.sqrt(v_psd_trimmed/i_psd_trimmed)/1e6
            
        # get peak power
        peak_power = np.max(np.abs(v_power)**2, axis=1)
        peak_power = peak_power[lower_idx:upper_idx]
        
        return impedance, peak_power, freq


    def get_spike_transfer(self, data, stim, freq_per_sec=1, window=0.25, max_freq=60, return_frequencies_per_time=False):
        """
        Detects spikes in a voltage signal and computes the number of spikes per frequency bin.
        
        Parameters
        ----------
        data : array-like
            The voltage signal (Vm) data, expected to be a one-dimensional array.
        freq_per_sec : int, optional
            The number of frequency bins to be considered per second for spike counting. Defaults to 1 bin/sec.
        window : float, optional
            Used to calculate fft
        max_freq: float, optional
            Maximum allowed frequency for stim
       return_frequencies_per_time: bool
           If True reutrn frequencies_per_time
        
        Returns
        -------
        spike_per_freq : list
            A list containing the count of spikes in each frequency bin.
        bins : array
            The edges of the frequency bins used for spike counting.
        max_power_freqs: array
            The frequency of max power at every window.
        """
        
        # get window in samples and find peak freq
        win = int(window*self.fs)
        f, t, power = stft(stim-np.mean(stim), fs=self.fs, nperseg=win, noverlap=0)
        power = np.abs(power)**2
        max_power_freqs = f[np.argmax(power, axis=0)][1:-1]
        fmin = np.min(max_power_freqs)
        fmax = np.max(max_power_freqs)
        
        if fmax > max_freq:
            print('Warning exceeded max freq')
            print(max_power_freqs)
        
        # get spike locations
        spike_locs, _ = find_peaks(data, prominence=self.prominence, distance=self.fs*self.dist/1000)

        # get spikes per frequency
        bins = np.arange(fmin, fmax+1, freq_per_sec)
        spike_per_freq , _ = np.histogram(spike_locs/self.fs, bins)
        
        # return max_power_freqs if needed
        if return_frequencies_per_time:
            return spike_per_freq, bins[1:], max_power_freqs

        return spike_per_freq, bins[1:]
        
        
    def analyze(self, data, stim, stim_type, wave=False, interpolation_factor=1,
        post_rheo_steps=-1, max_spikes_per_step=-1):
        """
        Unified feature extraction function for different stimulation protocols.
    
        Parameters
        ----------
        data : array
            Voltage trace from a single trial.
        stim : array
            Stimulation current trace.
        stim_type : str
            Type of stimulation protocol: 'io', 'rh', 'sch', or 'ch'.
        wave : bool, optional
            If True, extract spike waveforms (for 'io' only).
        interpolation_factor : int
            Interpolation factor for spike waveform (used if wave=True).
        post_rheo_steps : int
            Number of IO steps to analyze post-rheobase (default: -1 = all).
        max_spikes_per_step : int
            Max spikes per step to include in waveform (default: -1 = all).
    
        Returns
        -------
        dict
            Output depends on protocol:
            - 'io': spike frequency, input resistance, RMP, or waveform.
            - 'rh': rheobase.
            - 'sch': spike count per freq bin.
            - 'ch': impedance and peak power.
        """
        
        if stim_type == 'io':
            # Parse IO steps and detect spikes
            sig, amps, dur = self.parse_io(data, stim)
            locs = self.spike_locs(sig)
            
            # Return IO waveforms only
            if wave:
                wf, a2, t2 = self.select_waveforms(
                    sig, amps, locs,
                    interpolation_factor, post_rheo_steps, max_spikes_per_step
                )
                return {'mV': wf, 'amp': a2, 'time': t2}
            
            # Otherwise return IO metrics
            counts = np.array(self.count_spikes(locs))
            freqs = counts / (dur / self.fs)
            rin  = self.get_input_resistance(sig, amps, counts)
            rmp  = self.get_rmp(data)
            return {
                'spike_frequency': freqs,
                'input_resistance': rin,
                'amp': amps,
                'rmp': [rmp] * len(freqs)
            }
        
        # Get first location of first spike
        if stim_type == 'rh':
            return {'rheobase': [self.get_rheobase(data, stim)]}
        
        # Get chirp power freq range (2-59 Hz)
        if stim_type == 'ch':
            Z, P, f0 = self.get_chirp(data, stim, freq_range=(2,59))
            return {'impedance': Z, 'peak_power': P, 'freq': f0}
        
                
        if stim_type == 'sch':
            cnt, fr = self.get_spike_transfer(data, stim)
            return {'spike_count': cnt, 'freq': fr}
        
        return {}
        
    def save_validation_plot(self, data, stim, stim_type, save_path, wave=False,
                             post_rheo_steps=-1, max_spikes_per_step=-1):
        """
        Create and save a diagnostic plot of voltage and stimulus activity based on the stimulation protocol.
    
        Parameters
        ----------
        data : array
            Voltage signal (Vm).
        stim : array
            Stimulus signal (usually in pA).
        stim_type : str
            Type of stimulation protocol: 'io', 'rh', 'sch', or 'ch'.
        save_path : str
            Path to save the resulting plot.
        wave : bool
            If True and stim_type is 'io', highlights only selected spike waveforms.
        post_rheo_steps : int
            Number of IO steps post-rheobase to include when wave=True.
        max_spikes_per_step : int
            Max number of spikes per IO step to highlight when wave=True.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated matplotlib figure (also saved to disk).
        """
        t = np.arange(len(data)) / self.fs  # time axis in seconds
    
        if stim_type == 'io':
            # --- IO protocol: stack traces by step and mark spikes ---
            fig, ax = plt.subplots(figsize=(30, 15))
            sig, amps, dur = self.parse_io(data, stim)
            locs = self.spike_locs(sig)
    
            if wave:
                # Show only post-rheobase steps with selected spikes
                selected_idxs = []
                count = 0
                for i, l in enumerate(locs):
                    if count == post_rheo_steps:
                        break
                    if len(l) > 0:
                        selected_idxs.append(i)
                        count += 1
    
                for i in selected_idxs:
                    offset = i * 10
                    ax.plot(np.arange(len(sig[i])) / self.fs, sig[i] + offset, 'k-', alpha=0.7)
                    for j, loc in enumerate(locs[i]):
                        if max_spikes_per_step != -1 and j >= max_spikes_per_step:
                            break
                        ax.plot(loc / self.fs, sig[i][loc] + offset, 'rx', ms=20)
                ax.set_title("IO spikes (selected steps only)")
            else:
                # Plot all IO steps and all spikes
                for i, s in enumerate(sig):
                    offset = i * 10
                    ax.plot(np.arange(len(s)) / self.fs, s + offset, 'k-', alpha=0.7)
                    ax.plot(locs[i] / self.fs, s[locs[i]] + offset, 'rx', ms=20)
                ax.set_title("IO spikes (stacked steps)")
    
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (mV)")
            ax.legend(loc='upper right')
    
        else:
            # --- For 'rh', 'ch', 'sch': use dual-panel layout ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 15), sharex=True, height_ratios=[3, 1])
    
            if stim_type == 'rh':
                # Rheobase: plot spike and stimulus
                ax1.plot(t, data, 'k-')
                r0 = self.get_rheobase(data, stim)
                if r0 is not None:
                    idx = np.argmax(data > (np.median(data) + self.prominence))
                    ax1.axvline(idx / self.fs, color='r', ls='--')
                ax1.set_title("Rheobase spike")
                ax1.set_ylabel("Voltage (mV)")
    
                ax2.plot(t, stim, 'b-', alpha=0.7)
                ax2.set_ylabel("Stimulus (pA)")
                ax2.set_xlabel("Time (s)")
                ax2.set_title("Stimulus")
    
            elif stim_type == 'ch':
                # Chirp: show voltage and stimulus with frequency range
                ax1.plot(t, data, 'k-', label='Vm')
                _, _, f = self.get_chirp(data, stim)
                fmin, fmax = np.round(np.min(f), 1), np.round(np.max(f), 1)
                ax1.set_title(f"Chirp Protocol - Membrane Voltage (Freq range: {fmin}-{fmax} Hz)")
                ax1.set_ylabel("Voltage (mV)")
                ax1.legend(loc='upper right')
    
                ax2.plot(t, stim, 'b-', alpha=0.7)
                ax2.set_ylabel("Stimulus (pA)")
                ax2.set_xlabel("Time (s)")
                ax2.set_title("Chirp Protocol - Stimulus")
    
            elif stim_type == 'sch':
                # Short Chirp: plot spikes and stimulus frequency over time
                peaks, _ = find_peaks(data, prominence=self.prominence, distance=self.dist)
    
                # --- Top: Voltage trace with spikes ---
                ax1.plot(t, data, 'k-', label='Vm')
                ax1.plot(peaks / self.fs, data[peaks], 'rx', ms=10)
                for p in peaks:
                    ax1.axvline(p / self.fs, color='r', ls='--', lw=1, alpha=0.4)
                ax1.set_title("Short-chirp spikes")
                ax1.set_ylabel("Voltage (mV)")
    
                # --- Bottom: Stim frequency over time ---
                _, _, max_freqs = self.get_spike_transfer(data, stim, return_frequencies_per_time=True)
    
                # Interpolate frequency to full time resolution
                stim_freq = np.linspace(np.min(max_freqs), np.max(max_freqs), len(t))
                ax2.plot(t, stim_freq, 'k-', linewidth=2, label='Stim Freq')
    
                # Add vertical lines for spike timing
                for p in peaks:
                    ax2.axvline(p / self.fs, color='r', ls='--', lw=2, alpha=0.4)
    
                ax2.set_ylabel("Frequency (Hz)")
                ax2.set_xlabel("Time (s)")
                ax2.set_title("Stimulus frequency (interpolated)")
    
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return fig
    
