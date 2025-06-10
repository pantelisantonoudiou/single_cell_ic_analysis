# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##### ------------------------------------------------------------------- #####

def get_basic_properties(plot_data, group_cols):
    """
    Compute basic properties from IO data per group.

    Parameters
    ----------
    plot_data : pd.DataFrame
        DataFrame containing at least 'input_resistance' and 'rmp'.
    group_cols : list of str
        Columns to group by (e.g., ['treatment', 'projection', 'file'])

    Returns
    -------
    pd.DataFrame
        DataFrame with input resistance and RMP per group.
    """
    results = []
    for group_vals, df in plot_data.groupby(group_cols):
        ir = df['input_resistance'].mean() / 1e6
        rmp = df['rmp'].mean()
        props = dict(zip(group_cols, group_vals))
        props.update({
            'input_resistance': ir,
            'resting_membrane_potential': rmp
        })
        results.append(props)
    return pd.DataFrame(results)


def get_io_properties(plot_data, group_cols, show_plot=False):
    """
    Calculate IO curve features such as max firing rate, rheobase, slope, and frequency at different stimulus levels.

    Parameters
    ----------
    plot_data : pd.DataFrame
        Must include 'spike_frequency' and 'amp' columns.
    group_cols : list of str
        Columns to group by (e.g., ['treatment', 'cell_id']).
    show_plot : bool
        If True, plot the slope fit from rheobase to 90% max spike frequency.

    Returns
    -------
    pd.DataFrame
        Summary metrics for each group.
    """
    results = []

    for group_vals, df in plot_data.groupby(group_cols):
        df = df.sort_values('amp').reset_index(drop=True)

        max_fr = df['spike_frequency'].max()
        idx_half = (df['spike_frequency'] - max_fr / 2).abs().idxmin()
        w50 = df['amp'].loc[idx_half]

        # Firing rates at percent steps
        sfr_vals = {}
        for pct in [0.2, 0.4, 0.6, 0.8]:
            threshold = df['amp'].max() * pct
            amp_row = df[df['amp'] >= threshold]
            if not amp_row.empty:
                amp = amp_row['amp'].iloc[0]
                sfr_vals[f'fr_at_{int(pct * 100)}_percent_input'] = df[df['amp'] == amp]['spike_frequency'].values[0]
            else:
                sfr_vals[f'fr_at_{int(pct * 100)}_percent_input'] = np.nan

        fr_at_max_input = df['spike_frequency'].loc[df['amp'].idxmax()]
        rheobase = np.nan
        io_slope = np.nan

        # Rheobase: first amp with freq > 1 Hz
        above_thresh = df[df['spike_frequency'] > 1]
        if not above_thresh.empty:
            rheobase = above_thresh['amp'].values[0]

            # IO slope: linear fit from rheobase to 90% of max firing
            start_idx = above_thresh.index[0]
            ninety_pct_fr = 0.9 * max_fr
            try:
                end_idx = df[(df.index >= start_idx) & (df['spike_frequency'] >= ninety_pct_fr)].index[0]
                slope_segment = df.loc[start_idx:end_idx]

                # Ensure strictly increasing spike freq in segment
                y = slope_segment['spike_frequency'].values
                if len(y) >= 3 and np.all(np.diff(y) >= 0):
                    x = slope_segment['amp'].values
                    io_slope = np.polyfit(x, y, 1)[0]

                    if show_plot:
                        plt.figure()
                        plt.title("IO slope fit (rheobase to 90% max)")
                        plt.scatter(x, y, label="Data")
                        plt.plot(x, np.polyval(np.polyfit(x, y, 1), x), 'k-', lw=2, label="Fit")
                        plt.xlabel("Input current (pA)")
                        plt.ylabel("Firing rate (Hz)")
                        plt.legend()
                        plt.tight_layout()

            except IndexError:
                pass  # not enough high-frequency points to define slope

        props = dict(zip(group_cols, group_vals))
        props.update({
            'max_firing_rate': max_fr,
            'i_amp_at_half_max_fr': w50,
            'rheobase': rheobase,
            'io_slope': io_slope,
            'fr_at_max_input': fr_at_max_input
        })
        props.update(sfr_vals)
        results.append(props)

    return pd.DataFrame(results)


def get_waveform_properties(plot_data, group_cols, show_plot=False):
    """
    Extract average action potential waveform features per group (e.g., per cell).
    
    Parameters
    ----------
    plot_data : pd.DataFrame
        Must contain 'mV', 'time' columns for waveform data. Multiple waveforms per group will be averaged.
    group_cols : list of str
        Columns to group by (e.g., ['treatment', 'cell_id']).
    show_plot : bool
        If True, plot waveform and detected landmarks.

    Returns
    -------
    pd.DataFrame
        Waveform metrics calculated per group (1 row per group).
    """
    results = []

    for group_vals, df in plot_data.groupby(group_cols):
        
        
        # Compute mean waveform per group
        df = df.groupby('time')[['mV']].mean().reset_index()
        time = df['time'].values
        waveform = df['mV'].values
        step = time[1] - time[0]

        # Spike peak
        peak_idx = np.argmax(waveform)

        # AP threshold
        gradient = np.gradient(waveform[:peak_idx - 5])
        threshold_idx = np.where(gradient > 0.15)[0][0]
        ap_threshold = waveform[threshold_idx]
        ap_amp = waveform[peak_idx] - ap_threshold

        # AHP
        temp_wave = waveform[peak_idx : peak_idx + 200]
        ahp_idx = np.argmin(temp_wave) + peak_idx
        ahp = abs(ap_threshold - waveform[ahp_idx])

        # Peak to trough
        peak_to_trough = (ahp_idx - peak_idx) * step

        # Half-width
        half_amp = ap_amp / 2 + ap_threshold
        idx1 = threshold_idx + np.abs(waveform[threshold_idx:peak_idx] - half_amp).argmin()
        idx2 = peak_idx + np.abs(waveform[peak_idx:ahp_idx] - half_amp).argmin()
        half_width = (idx2 - idx1) * step

        # Rise time
        rise_time = (peak_idx - threshold_idx) * step

        if show_plot:
            plt.figure()
            plt.plot(time, waveform, lw=3, color='k')
            plt.plot(time[threshold_idx], waveform[threshold_idx], color='orange', marker='o', label='Threshold', ms=20)
            plt.plot(time[peak_idx], waveform[peak_idx], 'ro', label='Peak', ms=20)
            plt.plot(time[ahp_idx], waveform[ahp_idx], 'bo', label='AHP', ms=20)
            plt.plot(time[idx1], waveform[idx1], 'go', label='Half-width start', ms=20)
            plt.plot(time[idx2], waveform[idx2], 'go', label='Half-width end', ms=20)
            plt.title("Averaged AP waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (mV)")
            plt.legend()
            plt.tight_layout()

        props = dict(zip(group_cols, group_vals))
        props.update({
            'ap_peak': ap_amp,
            'threshold': ap_threshold,
            'ahp': ahp,
            'peak_to_trough': peak_to_trough,
            'rise_time': rise_time,
            'half_width': half_width
        })
        results.append(props)

    return pd.DataFrame(results)

