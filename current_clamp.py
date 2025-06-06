 
# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks, stft
import matplotlib.pyplot as plt
##### ------------------------------------------------------------------- #####


class Iclamp:
    """
    Detect and calculate spike properties for different current clamp stimulation protocols.
    """
    
    def __init__(self, fs, dist=1, spike_select=[2, 3], prominence=40, wlen=.1, show_io=False):
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
        self.show_io = show_io
            
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
        
        
        # plot detected stim
        if self.show_io:
            plt.figure()
            t = np.arange(len(stim))
            plt.plot(t, stim)
            plt.plot(t[start], stim[start], 'x')
            plt.plot(t[stop], stim[stop], 'o')

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
            if (a < 21) & (sc < 1):             
                # plt.figure()
                # plt.plot(s)
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
    
    
    def get_rmp(self, data, bins=np.arange(-100,0,.1)):
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
    
    def get_chirp(self, data, stim, stim_to_amp=1e12, data_to_v=1e3):
        """
        Calculate the impedance across a frequency range for a given voltage signal and input stimulus.
        
        Parameters
        ----------
        data : array, Voltage signal (one trial), usually representing neural activity or membrane voltage.
        stim : array, Input current signal (one trial), used to drive the system.
        stim_to_amp, float, conversion of stim to Amps
        data_to_v, covnersion of membrane voltage values to Volts
        
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
        lower_idx = np.where(f==2)[0][0]
        upper_idx = np.where(f==59)[0][0]+1
        v_psd_trimmed = v_psd[lower_idx:upper_idx]
        i_psd_trimmed = i_psd[lower_idx:upper_idx]
        freq = f[lower_idx:upper_idx]
        
        # get impedance
        impedance = np.sqrt(v_psd_trimmed/i_psd_trimmed)/1e6

        if self.show_io:
            plt.plot(freq, impedance, '-o')
            
        # get peak power
        peak_power = np.max(np.abs(v_power)**2, axis=1)
        peak_power = peak_power[lower_idx:upper_idx]
        
        return impedance, peak_power, freq
    
    def get_spike_transfer(self, data, stim, window):
        """
        Get number of spikes per frequency bin.
    
        Parameters
        ----------
        data : array, voltage signal (one trial)
        stim : array, input current (one trial)
        window : float, to calculate fft and detect spikes (seconds) 
    
        Returns
        -------
        spike_count : list, number of spikes per frequency
        freq, list, stim frequency per bin
        
        """
    
        # get window in samples and find peak freq
        win = int(window*self.fs)
        f, t, power = stft(stim-np.mean(stim), fs=self.fs, nperseg=win, noverlap=0)
        power = np.abs(power)**2
        freq = f[np.argmax(power, axis=0)]
        
        # get spike count per bin
        locs = self.spike_locs(data.reshape((-1, win)))
        spike_count = self.count_spikes(locs)
        
        return spike_count, freq[:-1]+int(1/window) # adjust window to match stim
        