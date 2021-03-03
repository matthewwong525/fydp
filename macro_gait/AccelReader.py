import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import datetime


class AccelReader():
    def __init__(self, accel_path, label='AccelReader', axis=None, orient_signal=True, low_pass=True, start=-1, end=-1, quiet=False):
        """
        AccelReader object reads accelerometer data from EDF files and loads it into the class

        Required Parameters:
        - `accel_path` (str): path to the accelerometer EDF
        - `label` (str): A label that's associated with the particular accelerometer data
        - `axis` (int): tells the EDF reader which column of the EDF to access

        Optional Parameters:
        - `start` (int): starting datapoint to splice data
        - `end` (int): ending datapoint to splice data
        - `quiet` (bool): stops printing
        """
        if not quiet:
            print("%s: Reading Accel Data..." % label)
        self.accel_path = accel_path
        self.freq, self.data, self.xz_data, self.timestamps, self.axis = AccelReader.get_accel_csv_data(
            path=accel_path, axis=axis, start=start, end=end)
        self.raw_data = self.data
        if not (start > -1 and end > -1):
            self.start_dp = 0
            self.end_dp = len(self.data)
        else:
            self.start_dp = start
            self.end_dp = end

        self.label = label
        self.quiet = quiet
        self.sig_length = len(self.data)

        if orient_signal:
            self.flip_signal()
        if low_pass:
            self.lowpass_filter()

    @staticmethod
    def find_dp(path, timestamp_str, length, axis=1):
        """
        Gets start and end time based on a timestamp and length(# data points)
        """
        accel_file = pyedflib.EdfReader(path)
        time_delta = pd.to_timedelta(
            1 / accel_file.getSampleFrequency(axis), unit='s')
        start = int((pd.to_datetime(timestamp_str) -
                     accel_file.getStartdatetime()) / time_delta)
        end = int(start + pd.to_timedelta(length, unit='s') / time_delta)
        accel_file.close()
        return start, end

    
    @staticmethod
    def get_accel_csv_data(path, axis=None, start=-1, end=-1):
        df = pd.read_csv(path)
        timestamps = pd.to_datetime(df['timestamps'])
        freq = df.shape[0] / (timestamps.max() - timestamps.min()).total_seconds()
        # finds most active axis and sets it as walking
        other_axes = ['ax', 'ay', 'az']
        if not axis:
            axis = df[other_axes].abs().sum().idxmax(axis=1)
            other_axes.remove(axis)
        timestamps = timestamps.to_numpy()
        data = df[axis].to_numpy()
        xz_data = np.sqrt((df[other_axes] ** 2).sum(axis=1))
        return freq, data, xz_data, timestamps, axis

    def flip_signal(self):
        """
        Finds orientation based on lowpassed signal and flips the signal
        """

        cutoff_freq = self.freq * 0.005
        sos = signal.butter(N=1, Wn=cutoff_freq,
                               btype='low', fs=self.freq, output='sos')
        orientation = signal.sosfilt(sos, self.data)
        flip_ind = np.where(orientation < -0.25)
        self.orientation = orientation
        self.data[flip_ind] = -self.data[flip_ind]

    # 40Hz butter low pass filter
    def lowpass_filter(self, order=2, cutoff_ratio=0.17):
        """
        Applies a lowpass filter on the accelerometer data
        """
        cutoff_freq = self.freq * cutoff_ratio
        sos = signal.butter(N=order, Wn=cutoff_freq,
                               btype='low', fs=self.freq, output='sos')
        self.data = signal.sosfilt(sos, self.data)

    def plot(self, return_plt=False):
        """
        Plots the signal, with additional option to plot other signals that are of the same length
        """
        dp_range = np.arange(self.start_dp, self.end_dp)
        ax1 = plt.subplot(311)
        ax1.set_title('Raw Accelerometer Data')
        plt.plot(dp_range, self.raw_data, 'r-')
        plt.grid(True)

        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_title('Low Pass Filtered/Flipped Signal')
        plt.plot(dp_range, self.data, 'r-')
        plt.grid(True)

        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_title('Device Orientation')
        # filtered_legend = ['Flipped Signal', 'Original Signal']
        # ax3.set_yticks(np.arange(len(filtered_legend)))
        # ax3.set_yticklabels(filtered_legend)
        plt.plot(dp_range, self.orientation, 'r-')
        plt.grid(True)
        plt.tight_layout()

        if not return_plt:
            plt.show()
        else:
            return plt

    @staticmethod
    def plot_sigs(start, end, signals):
        """
        Plots the signal, with additional option to plot other signals that are of the same length
        Maybe check if all signals are of the same length?
        """
        time_range = np.arange(start, end)

        fig, axs = plt.subplots(len(signals), sharex=True)
        axs = [axs] if len(signals) == 1 else axs

        for i, ax in enumerate(axs):
            ax.plot(time_range, signals[i][start:end], 'r-')

        plt.show()

if __name__ == '__main__':
    path = '/Users/matthewwong/Documents/coding/fydp/raspberrypi_data.csv'
    ac = AccelReader(path)
    
