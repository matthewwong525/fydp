import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import signal
from tqdm.auto import tqdm
from macro_gait.AccelReader import AccelReader


class StepDetection(AccelReader):
    # increase swing threshold
    # decrease heel strike length
    push_off_threshold = 0.85
    swing_threshold = 0.5
    heel_strike_threshold = -5
    pushoff_time = 0.4
    swing_down_detect_time = 0.3
    swing_up_detect_time = 0.1
    swing_phase_time = swing_down_detect_time + swing_up_detect_time * 2
    heel_strike_detect_time = 0.8
    foot_down_time = 0.1

    def __init__(self, pushoff_df=None, label='StepDetection', **kwargs):
        """
        StepDetectioon class performs step detection through a steady state controller algorithm

        Required Parameters:
        accel_reader_obj (AccelReader): Object from AccelReader class
        pushoff_df (pandas.DataFrame): A dataframe that outlines the mean, std, and min/max for a pushoff

        Optional Parameters:
        - `quiet` (bool): stops printing
        """
        super(StepDetection, self).__init__(**kwargs)
        self.label = label
        self.pushoff_len = int(self.pushoff_time * self.freq)
        self.states = {1: 'stance', 2: 'push-off',
                       3: 'swing-up', 4: 'swing-down', 5: 'footdown'}
        self.state = self.states[1]
        self.pushoff_df  = StepDetection.get_pushoff_stats(self.accel_path, start_end_times=[(self.start_dp, self.end_dp)], quiet=self.quiet) if pushoff_df is None else pushoff_df
        self.swing_peaks = []

        # set threshold values
        if {'swing_down_mean', 'swing_down_std', 'swing_up_mean', 'swing_up_std', 'heel_strike_mean', 'heel_strike_std'}.issubset(self.pushoff_df.columns):
            self.swing_phase_time = self.pushoff_df['swing_down_mean'].iloc[0] + self.pushoff_df['swing_down_std'].iloc[0] + self.pushoff_df['swing_up_mean'].iloc[0] + self.pushoff_df['swing_up_std'].iloc[0]
            self.heel_strike_detect_time = 0.5 + self.pushoff_df['swing_up_mean'].iloc[0] + 2 * self.pushoff_df['swing_up_std'].iloc[0]
            self.heel_strike_threshold = -3 - self.pushoff_df['heel_strike_mean'].iloc[0]/(2 * self.heel_strike_threshold)

        self.step_detect()

    @staticmethod
    def get_pushoff_stats(accel_path, start_end_times=[(-1, -1)], axis=None, quiet=True):
        pushoff_sig_list = []
        swingdown_times = []
        swingup_times = []
        heelstrike_values = []

        for start, end in start_end_times:
            step = StepDetection(accel_path=accel_path, label='get_pushoff_stats',
                                axis=axis, start=start, end=end, quiet=True, pushoff_df=pd.read_csv('pushoff_OND07_left.csv'))
            pushoff_sig = StepDetection.get_pushoff_sigs(step, quiet=quiet)
            step_summary = step.export_steps()
            toe_offs = step_summary.loc[step_summary['step_state'] == 'success', 'swing_start_time']
            mid_swings = step_summary.loc[step_summary['step_state'] == 'success', 'mid_swing_time']
            heel_strikes = step_summary.loc[step_summary['step_state'] == 'success', 'heel_strike_time']
            step_indices = step_summary.loc[step_summary['step_state'] == 'success', 'step_index'] - step.start_dp

            mid_swing_indices = step_indices + (step.pushoff_time + (mid_swings - toe_offs).dt.total_seconds()) * step.freq

            if len(pushoff_sig) == 0:
                print('WARNING: No steps found %s (start=%s, end=%s)' % (os.path.basename(accel_path), str(start), str(end)))
                continue
            pushoff_sig_list.append(pushoff_sig)
            swingdown_times.append((mid_swings - toe_offs).dt.total_seconds())
            swingup_times.append((heel_strikes - mid_swings).dt.total_seconds())
            heelstrike_values.append([np.min(step.heel_strike_detect(int(ms_ind))) for ms_ind in mid_swing_indices]) 
 
        if len(pushoff_sig_list) == 0:
            return None

        pushoff_sig_list = np.concatenate(pushoff_sig_list)
        swingdown_times = np.concatenate(swingdown_times)
        swingup_times = np.concatenate(swingup_times)
        heelstrike_values = np.concatenate(heelstrike_values)
        po_avg_sig = np.mean(pushoff_sig_list, axis=0)
        po_std_sig = np.std(pushoff_sig_list, axis=0)
        po_max_sig = np.max(pushoff_sig_list, axis=0)
        po_min_sig = np.min(pushoff_sig_list, axis=0)

        sdown_mean = np.nanmean(swingdown_times)
        sdown_std = np.nanstd(swingdown_times)
        sup_mean = np.nanmean(swingup_times)
        sup_std = np.nanstd(swingup_times)
        hs_mean = np.nanmean(sorted(heelstrike_values, reverse=True)[:len(heelstrike_values)//4])
        hs_std = np.nanstd(sorted(heelstrike_values, reverse=True)[:len(heelstrike_values)//4])

        if len(pushoff_sig_list) < 20:
            print('WARNING: less than 20 steps used for pushoff DF for %s' % os.path.basename(accel_path))

        pushoff_df = pd.DataFrame(
            {'avg': po_avg_sig, 'std': po_std_sig, 'max': po_max_sig, 'min': po_min_sig,
             'swing_down_mean': sdown_mean, 'swing_down_std': sdown_std,
             'swing_up_mean': sup_mean, 'swing_up_std': sup_std,
             'heel_strike_mean': hs_mean, 'heel_strike_std': hs_std})
        
        return pushoff_df

        # TODO: add right peaks to the detection
        # 1. find avg push_off signal

    @classmethod
    def get_pushoff_sigs(cls, step_obj, peaks=None, quiet=False):
        """
        Creates average pushoff dataframe that is used to find pushoff data
        """
        pushoff_sig_list = []
        pushoff_len = step_obj.pushoff_time * step_obj.freq

        if not peaks:
            peaks = np.array(step_obj.step_indices) + pushoff_len
        for i, peak_ind in tqdm(enumerate(peaks), desc="Generating pushoff average", total=len(peaks), disable=quiet):
            pushoff_sig = step_obj.data[int(peak_ind - pushoff_len):int(peak_ind)]
            pushoff_sig_list.append(pushoff_sig)

        return np.array(pushoff_sig_list)

    @staticmethod
    def df_from_csv(csv_file):
        """
        reads a csv file and returns a DataFrame object
        """
        return pd.read_csv(csv_file)

    def push_off_detection(self):
        """
        Detects the steps based on the pushoff_df
        """
        if not self.quiet:
            print('%s: Finding Indices for pushoff' % self.label)

        pushoff_avg = self.pushoff_df['avg']

        cc_list = StepDetection.window_correlate(self.data, pushoff_avg)

        # TODO: DISTANCE CAN BE ADJUSTED FOR THE LENGTH OF ONE STEP RIGHT NOW ASSUMPTION IS THAT A PERSON CANT TAKE 2 STEPS WITHIN 0.5s
        pushoff_ind, _ = signal.find_peaks(
            cc_list, height=self.push_off_threshold, distance=0.2 * self.freq)

        return pushoff_ind

    def step_detect(self):
        """
        Detects the steps within the accelerometer data. Based on this paper:
        https://ris.utwente.nl/ws/portalfiles/portal/6643607/00064463.pdf
        """
        pushoff_ind = self.push_off_detection()
        end_pushoff_ind = pushoff_ind + self.pushoff_len
        state_arr = np.zeros(self.data.size)
        detects = {'push_offs': len(end_pushoff_ind), 'mid_swing_peak': [], 'swing_up': [], 'swing_down': [
        ], 'heel_strike': [], 'next_i': [], 'pushoff_mean': []}
        detect_arr = np.zeros(self.data.size)

        end_i = None
        step_indices = []
        step_lengths = []

        for count, i in tqdm(enumerate(end_pushoff_ind), disable=self.quiet, total=len(end_pushoff_ind), desc='%s: Step Detection' % self.label):
            # check if next index within the previous detection
            if end_i and i - self.pushoff_len < end_i:
                detects['next_i'].append(i - 1)
                continue

            # mean/std check for pushoff, state = 1
            pushoff_mean = np.mean(self.data[i - self.pushoff_len:i])
            upper = (self.pushoff_df['avg'] + 3 * self.pushoff_df['std'])
            lower = (self.pushoff_df['avg'] - 3 * self.pushoff_df['std'])
            if not np.any((pushoff_mean < upper) & (pushoff_mean > lower)):
                detects['pushoff_mean'].append(i - 1)
                continue

            # midswing peak detection
            mid_swing_i = self.mid_swing_peak_detect(i)
            if mid_swing_i is None:
                detects['mid_swing_peak'].append(i - 1)
                continue

            # swing down, state = 2
            # sdown_cc, sup_cc = self.swing_detect(i, mid_swing_i)
            # if not max(sdown_cc) > self.swing_threshold:
            #     detects['swing_down'].append(i - 1)
            #     continue

            # # swing up, state = 3
            # if not max(sup_cc) > self.swing_threshold:
            #     detects['swing_up'].append(i - 1)
            #     continue

            # heel-strike, state = 4
            accel_derivatives = self.heel_strike_detect(mid_swing_i)
            accel_threshold_list = np.where(
                accel_derivatives < self.heel_strike_threshold)[0]
            if len(accel_threshold_list) == 0:
                detects['heel_strike'].append(i - 1)
                continue
            accel_ind = accel_threshold_list[0] + mid_swing_i
            end_i = accel_ind + int(self.foot_down_time * self.freq)

            state_arr[i - self.pushoff_len:i] = 1
            state_arr[i:mid_swing_i] = 2
            state_arr[mid_swing_i:accel_ind] = 3
            state_arr[accel_ind:end_i] = 4

            step_indices.append(i - self.pushoff_len)
            step_lengths.append(end_i - (i - self.pushoff_len))

        detect_arr[detects['swing_down']] = 1
        detect_arr[detects['swing_up']] = 2
        detect_arr[detects['heel_strike']] = 3
        detect_arr[detects['next_i']] = 4
        detect_arr[detects['pushoff_mean']] = 5
        detect_arr[detects['mid_swing_peak']] = 6

        self.state_arr = state_arr
        self.step_indices = step_indices
        self.step_lengths = step_lengths
        self.detect_arr = detect_arr

        return state_arr, step_indices

    @staticmethod
    def window_correlate(sig1, sig2):
        """
        Does cross-correlation between 2 signals over a window of indices
        """
        sig = np.array(max([sig1, sig2], key=len))
        window = np.array(min([sig1, sig2], key=len))

        engine = 'cython' if len(sig) < 100000 else 'numba'
        cc = pd.Series(sig
                       ).rolling(window=len(window)
                                 ).apply(lambda x: np.corrcoef(x, window)[0, 1], raw=True, engine=engine
                                         ).shift(-len(window) + 1
                                                 ).fillna(0
                                                          ).to_numpy()

        return cc

    def mid_swing_peak_detect(self, pushoff_ind):
        swing_detect = int(self.freq * self.swing_phase_time)  # length to check for swing
        detect_window = self.data[pushoff_ind:pushoff_ind + swing_detect]
        peaks, prop = signal.find_peaks(-detect_window,
                                        distance=swing_detect * 0.25,
                                        prominence=0.2, wlen=swing_detect,
                                        width=[0 * self.freq, self.swing_phase_time * self.freq], rel_height=0.75)
        results  = signal.peak_widths(-detect_window, peaks)
        prop['widths'] = results[0]
        if len(peaks) == 0:
            return None

        return pushoff_ind + peaks[np.argmax(prop['widths'])]

    def swing_detect(self, pushoff_ind, mid_swing_ind):
        """
        Detects swings (either up or down) given a starting index (window_ind).
        Swing duration is preset.
        """
        # swinging down
        detect_window = self.data[pushoff_ind:mid_swing_ind]
        swing_len = mid_swing_ind - pushoff_ind
        swing_down_sig = -np.arange(swing_len) + swing_len / 2 + np.mean(detect_window)

        # swinging up
        swing_up_detect = int(self.freq * self.swing_up_detect_time)  # length to check for swing
        swing_up_detect_window = self.data[mid_swing_ind:mid_swing_ind + swing_up_detect]
        swing_up_sig = -(-np.arange(swing_up_detect) + swing_up_detect / 2 + np.mean(detect_window))

        swing_down_cc = [np.corrcoef(detect_window, swing_down_sig)[0, 1]] if detect_window.shape[0] > 1 else [0] 
        swing_up_cc = [np.corrcoef(swing_up_detect_window, swing_up_sig)[0, 1]] if swing_up_detect_window.shape[0] > 1 else [0]

        return (swing_down_cc, swing_up_cc)

    def heel_strike_detect(self, window_ind):
        """
        Detects a heel strike based on the change in acceleration over time.
        """
        heel_detect = int(self.freq * self.heel_strike_detect_time)
        detect_window = self.data[window_ind:window_ind + heel_detect]
        accel_t_plus1 = np.append(
            detect_window[1:detect_window.size], detect_window[-1])
        accel_t_minus1 = np.insert(detect_window[:-1], 0, detect_window[0])
        accel_derivative = (accel_t_plus1 - accel_t_minus1) / (2 / self.freq)

        return accel_derivative

    def export_steps(self):
        assert len(self.detect_arr) == len(self.timestamps)
        failed_step_indices = np.where(self.detect_arr > 0)[0]
        failed_step_timestamps = self.timestamps[failed_step_indices]

        error_mapping = {1: 'swing_down', 2: 'swing_up',
                         3: 'heel_strike_too_small', 4: 'too_close_to_next_i',
                         5: 'too_far_from_pushoff_mean', 6: 'mid_swing_peak_not_detected'}
        failed_step_state = list(map(error_mapping.get, self.detect_arr[failed_step_indices]))

        step_timestamps = self.timestamps[self.step_indices]

        swing_start = np.where((self.state_arr == 1) & (np.roll(self.state_arr, -1) == 2))[0]
        mid_swing = np.where((self.state_arr == 2) & (np.roll(self.state_arr, -1) == 3))[0]
        heel_strike = np.where((self.state_arr == 3) & (np.roll(self.state_arr, -1) == 4))[0]

        pushoff_start = swing_start - int(self.pushoff_time * self.freq)
        gait_cycle_end = heel_strike + int(self.foot_down_time * self.freq)
        step_times = (gait_cycle_end - pushoff_start) / self.freq
        avg_speed = [np.mean(self.xz_data[i:i + int(lengths * self.freq)]) * 9.81 * lengths for i, lengths in zip(self.step_indices, step_times)]

        assert len(self.step_indices) == len(swing_start)
        assert len(self.step_indices) == len(mid_swing)
        assert len(self.step_indices) == len(heel_strike)

        successful_steps = pd.DataFrame({
            'step_time': step_timestamps,
            'step_index': np.array(self.step_indices) + self.start_dp,
            'step_state': 'success',
            'swing_start_time': self.timestamps[swing_start],
            'mid_swing_time': self.timestamps[mid_swing],
            'heel_strike_time': self.timestamps[heel_strike],
            'foot_down_time': self.timestamps[heel_strike + int(self.foot_down_time * self.freq)],
            'swing_start_accel': self.data[swing_start],
            'mid_swing_accel': self.data[mid_swing],
            'heel_strike_accel': self.data[heel_strike],
            'step_length_sec': step_times,
            'avg_speed': avg_speed
        })
        failed_steps = pd.DataFrame({
            'step_time': failed_step_timestamps,
            'step_index': np.array(failed_step_indices) + self.start_dp,
            'step_state': failed_step_state
        })
        df = pd.concat([successful_steps, failed_steps], sort=True)
        df = df.sort_values(by='step_index')
        df = df.reset_index(drop=True)

        return df

    def plot(self, return_plt=False):
        """
        Plots the accelerometer data, the states detected, and the detected pushoffs that were eliminated
        """

        dp_range = np.arange(self.start_dp, self.end_dp)

        ax1 = plt.subplot(311)
        ax1.set_title('Accelerometer Data')
        plt.plot(dp_range, self.data, 'r-')
        #plt.plot(dp_range[self.swing_peaks], self.data[self.swing_peaks], 'bo')
        plt.grid(True)

        ax2 = plt.subplot(312, sharex=ax1)
        states_legend = ['stance', 'pushoff',
                         'swing down', 'swing up', 'heel strike']
        ax2.set_title('States of Steps in Accelerometer Data')
        ax2.set_yticks(np.arange(len(states_legend)))
        ax2.set_yticklabels(states_legend)
        # ax2.legend([0,1,2,3,4], ['stance', 'pushoff', 'swing down', 'swing up', 'heel strike'])
        plt.plot(dp_range, self.state_arr, "b-")
        plt.grid(True)

        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_title('Push off signals filtered out by SSC')
        filtered_legend = ['', 'swing down', 'swing up', 'heel strike',
                           'next pushoff too close', 'pushoff mean too far', 'mid_swing_peak']
        ax3.set_yticks(np.arange(len(filtered_legend)))
        ax3.set_yticklabels(filtered_legend)
        plt.plot(dp_range, self.detect_arr, "go")
        plt.grid(True)

        plt.tight_layout()
        if not return_plt:
            plt.show()
        else:
            return plt

if __name__ == '__main__':
    path = '/Users/matthewwong/Documents/coding/fydp/raspberrypi-1_data.csv'
    pushoff_df = '/Users/matthewwong/Documents/coding/fydp/macro_gait/pushoff_OND07_left.csv'
    obj = StepDetection(accel_path=path, pushoff_df=pd.read_csv(pushoff_df), label='hello')
    
    print('completed!')
