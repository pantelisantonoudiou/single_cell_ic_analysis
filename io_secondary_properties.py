# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
from scipy import interpolate
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
    Calculate IO curve features like slope, rheobase, etc.

    Parameters
    ----------
    plot_data : pd.DataFrame
        Must include 'spike_frequency' and 'amp'.
    group_cols : list of str
        Grouping columns.
    show_plot : bool
        Whether to show fitting plots.

    Returns
    -------
    pd.DataFrame
        IO curve metrics per group.
    """
    results = []
    for group_vals, df in plot_data.groupby(group_cols):
        df = df.sort_values('amp')
        max_fr = df['spike_frequency'].max()
        idx = (df['spike_frequency'] - max_fr / 2).abs().idxmin()
        w50 = df['amp'].loc[idx]

        sfr_vals = {}
        for pct in [0.2, 0.4, 0.6, 0.8]:
            threshold = df['amp'].max() * pct
            amp = df[df['amp'] >= threshold]['amp'].iloc[0]
            sfr_vals[f'fr_at_{int(pct*100)}_percent_input'] = df[df['amp'] == amp]['spike_frequency'].values[0]

        fr_at_max_input = df['spike_frequency'].loc[df['amp'].idxmax()]
        rheobase = np.nan
        io_slope = np.nan

        above_threshold = df[df['spike_frequency'] > 1]
        if not above_threshold.empty:
            rheobase = above_threshold['amp'].values[0]
            x_min = above_threshold.index[0]
            x_max = df['spike_frequency'].idxmax()
            slope_region = df.loc[x_min:x_max]
            keep = slope_region['spike_frequency'] < max_fr * 0.9
            if keep.sum() >= 3:
                x = slope_region.loc[keep, 'amp'].values
                y = slope_region.loc[keep, 'spike_frequency'].values
                io_slope = np.polyfit(x, y, 1)[0]

                if show_plot:
                    plt.scatter(x, y)
                    plt.plot(x, np.polyval(np.polyfit(x, y, 1), x), 'r-')

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
    Extract action potential waveform features per group.

    Parameters
    ----------
    plot_data : pd.DataFrame
        Must contain 'mV' and 'time' columns for waveform data.
    group_cols : list of str
        Columns to group by (e.g., cell_id, treatment).
    show_plot : bool
        If True, show plots for visual inspection of AP landmarks.

    Returns
    -------
    pd.DataFrame
        Waveform metrics calculated per group.
    """
    results = []

    for group_vals, df in plot_data.groupby(group_cols):
        # Extract waveform and time
        waveform = df['mV'].values
        time = df['time'].values
        step = time[1] - time[0]  # Sampling interval

        # Identify peak of action potential
        peak_idx = np.argmax(waveform)

        # Estimate AP threshold: first steep rise before peak
        gradient = np.gradient(waveform[:peak_idx - 5])
        threshold_idx = np.where(gradient > 0.15)[0][0]
        ap_threshold = waveform[threshold_idx]
        ap_amp = waveform[peak_idx] - ap_threshold

        # After-hyperpolarization (AHP): minimum after peak
        temp_wave = waveform[peak_idx : peak_idx + 200]
        ahp_idx = np.argmin(temp_wave) + peak_idx
        ahp = abs(ap_threshold - waveform[ahp_idx])

        # Peak-to-trough time
        peak_to_trough = (ahp_idx - peak_idx) * step

        # Half-width: time between rise and fall at half-max
        half_amp = ap_amp / 2 + ap_threshold
        idx1 = threshold_idx + np.abs(waveform[threshold_idx:peak_idx] - half_amp).argmin()
        idx2 = peak_idx + np.abs(waveform[peak_idx:ahp_idx] - half_amp).argmin()
        half_width = (idx2 - idx1) * step

        # Rise time: from threshold to peak
        rise_time = (peak_idx - threshold_idx) * step

        # Optional diagnostic plot
        if show_plot:
            plt.figure()
            plt.plot(time, waveform)
            plt.plot(time[threshold_idx], waveform[threshold_idx], 'kx', label='Threshold')
            plt.plot(time[peak_idx], waveform[peak_idx], 'rx', label='Peak')
            plt.plot(time[ahp_idx], waveform[ahp_idx], 'bx', label='AHP')
            plt.plot(time[idx1], waveform[idx1], 'gx', label='Half-width start')
            plt.plot(time[idx2], waveform[idx2], 'gx', label='Half-width end')
            plt.legend()
            plt.title(f"Waveform features: {dict(zip(group_cols, group_vals))}")
            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage (mV)")
            plt.tight_layout()
            plt.show()

        # Package metrics
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
