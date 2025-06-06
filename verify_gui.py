# ---------------------- IMPORTS ---------------------- #
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtCore
import matplotlib
import adi
from scipy.signal import find_peaks

matplotlib.rcParams.update({'font.size': 20})
# ----------------------------------------------------- #


class matplotGui:
    """
    Interactive Matplotlib GUI to verify spike detection thresholds from ADI files.

    This class walks through individual stimulation events (e.g., 'io' blocks),
    allowing the user to visually verify and adjust the spike detection threshold
    (prominence) per recording file.

    Parameters
    ----------
    index_df : pd.DataFrame
        DataFrame containing at least the following columns:
        ['full_path', 'file_name', 'stim_type', 'block', 'start_sample', 'stop_sample'].
    data_ch : int
        Index of the ADI voltage channel to load for plotting.
    prominence : float
        Default spike detection prominence value.

    Attributes
    ----------
    result_df : pd.DataFrame
        Populated after GUI is closed. Contains 'file_name', 'threshold', and 'accepted'.
    """

    ind = 0  # internal counter

    def __init__(self, index_df, data_ch=0, prominence=30):
        self.index_df = index_df.copy()
        self.data_ch = data_ch
        self.result_df = None  # will be assigned after GUI ends

        # Initialize columns
        if 'accepted' not in self.index_df.columns:
            self.index_df.insert(0, 'accepted', -1)
            self.index_df.insert(0, 'threshold', prominence)
        else:
            self.index_df['threshold'] = prominence

        self.wait_time = 0.05
        self.bcg_color = {-1: 'w', 0: 'salmon', 1: 'palegreen'}

        # Initialize plot
        self.fig, self.axs = plt.subplots(1, 1, figsize=(25, 15))
        self.axs.spines["top"].set_visible(False)
        self.axs.spines["right"].set_visible(False)

        self.plot_data()

        # Instructions
        plt.subplots_adjust(bottom=0.15)
        self.fig.text(
            0.5, 0.04,
            '** Accept/Reject = a/r,     ←/→ = Prev/Next,     ↑/↓ = Threshold,     Esc = Exit **',
            ha="center",
            bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9))
        )

        # Connect events
        self.fig.canvas.callbacks.connect('key_press_event', self.keypress)
        self.fig.canvas.callbacks.connect('close_event', self.close_event)

        # Disable close (X) button
        win = plt.gcf().canvas.manager.window
        win.setWindowFlags(win.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        win.setWindowFlags(win.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)

        # Show GUI and block execution
        plt.show()
        self.result_df = self.index_df.copy()  # Store updated DataFrame after close

    def get_index(self):
        if self.ind >= len(self.index_df):
            self.ind = 0
        elif self.ind < 0:
            self.ind = len(self.index_df) - 1
        self.i = self.ind

    def set_background_color(self):
        clr = self.bcg_color[self.index_df['accepted'][self.i]]
        self.axs.set_facecolor(clr)

    def load_data(self):
        self.get_index()
        row = self.index_df.loc[self.ind]

        fread = adi.read_file(row['full_path'])
        ch = fread.channels[self.data_ch]
        fs = int(ch.fs[0])

        start = int(row['start_sample'])
        stop = int(row['stop_sample']) if pd.notnull(row['stop_sample']) else None
        if start < 1:
            start = 1

        data = ch.get_data(row['block'] + 1, start_sample=start, stop_sample=stop)
        return data, fs, os.path.basename(row['full_path'])

    def plot_data(self):
        self.axs.clear()
        try:
            raw_data, fs, file_name = self.load_data()
        except Exception as e:
            print(f"Error loading data for index {self.ind}: {e}")
            self.index_df.at[self.ind, 'accepted'] = 0
            return

        spike_amp = self.index_df['threshold'][self.ind]
        spike_locs, _ = find_peaks(raw_data, prominence=spike_amp, distance=fs * 1e-3, wlen=int(fs / 10))

        t = np.arange(raw_data.shape[0]) / fs
        threshold_str = f"Threshold = {spike_amp}"

        self.axs.plot(t, raw_data, color='black', label=threshold_str)
        self.axs.plot(t[spike_locs], raw_data[spike_locs], 'o', color='darkmagenta')
        self.axs.legend(loc='upper right')

        title = f"{self.ind + 1} of {len(self.index_df)} | {file_name}"
        self.fig.suptitle(title, fontsize=22)
        self.set_background_color()
        self.axs.set_xlabel('Time (s)')
        self.axs.set_ylabel('Vm (mV)')
        self.fig.canvas.draw()

    def close_event(self, event):
        plt.close()

    def keypress(self, event):
        if event.key == 'right':
            self.ind += 1
            self.plot_data()

        elif event.key == 'left':
            self.ind -= 1
            self.plot_data()

        elif event.key == 'up':
            self.index_df.at[self.ind, 'threshold'] += 5
            self.plot_data()

        elif event.key == 'down':
            self.index_df.at[self.ind, 'threshold'] -= 5
            self.plot_data()

        elif event.key == 'a':
            self.index_df.at[self.i, 'accepted'] = 1
            self.set_background_color()
            plt.draw(); plt.pause(self.wait_time)
            self.ind += 1
            self.plot_data()

        elif event.key == 'r':
            self.index_df.at[self.i, 'accepted'] = 0
            self.set_background_color()
            plt.draw(); plt.pause(self.wait_time)
            self.ind += 1
            self.plot_data()

        elif event.key == 'ctrl+a':
            self.index_df['accepted'] = 1
            self.set_background_color()
            plt.draw(); plt.pause(self.wait_time)

        elif event.key == 'ctrl+r':
            self.index_df['accepted'] = 0
            self.set_background_color()
            plt.draw(); plt.pause(self.wait_time)

        elif event.key == 'enter':
            plt.close()

    def get_result(self):
        """
        Returns
        -------
        pd.DataFrame
            DataFrame containing all verification results.
        """
        return self.result_df
