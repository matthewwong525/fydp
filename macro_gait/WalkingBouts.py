import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from macro_gait.StepDetection import StepDetection
from macro_gait.AccelReader import AccelReader
from macro_gait.SensorFusion import SensorFusion

class WalkingBouts():
    def __init__(self, left_accel_path, right_accel_path, start_time=None, duration_sec=None, bout_num_df=None, legacy_alg=False, left_kwargs={}, right_kwargs={}):
        """
        WalkingBouts class finds the bouts within two StepDetection objects

        Required Parameters:
        `left_stepdetector` (StepDetection): left StepDetection Class
        `right_stepdetector` (StepDetection): Right StepDetection Class
        """
        # helps synchronize both bouts
        if start_time and duration_sec:
            l_start, l_end = AccelReader.find_dp(left_accel_path, start_time, duration_sec)
            r_start, r_end = AccelReader.find_dp(right_accel_path, start_time, duration_sec)
            left_kwargs['start'], left_kwargs['end'] = l_start, l_end
            right_kwargs['start'], right_kwargs['end'] = r_start, r_end

        left_stepdetector = SensorFusion(path=left_accel_path, **left_kwargs)
        right_stepdetector = SensorFusion(path=right_accel_path, **right_kwargs)
        self.left_step_df = left_stepdetector.export_steps()
        self.right_step_df = right_stepdetector.export_steps()
        self.left_step_df['step_time'] = pd.to_datetime(self.left_step_df['step_time'])
        self.right_step_df['step_time'] = pd.to_datetime(self.right_step_df['step_time'])
        self.left_step_df['foot'] = 'left'
        self.right_step_df['foot'] = 'right'

        self.left_states = left_stepdetector.state_arr
        self.right_states = right_stepdetector.state_arr
        self.left_steps_failed = left_stepdetector.detect_arr
        self.right_steps_failed = right_stepdetector.detect_arr
        self.freq = left_stepdetector.freq  # TODO: check if frequencies are the same
        # assert left_stepdetector.freq == right_stepdetector.freq
        if legacy_alg:
            self.bout_num_df = WalkingBouts.identify_bouts(left_stepdetector, right_stepdetector) if bout_num_df is None else bout_num_df
        else:
            left_bouts = WalkingBouts.identify_bouts_one(left_stepdetector)
            right_bouts = WalkingBouts.identify_bouts_one(right_stepdetector)
            self.bout_num_df = WalkingBouts.find_overlapping_times(left_bouts, right_bouts)
            
        self.sig_length = min(left_stepdetector.sig_length, right_stepdetector.sig_length)
        self.left_data = left_stepdetector.data
        self.right_data = right_stepdetector.data
        self.start_dp = left_stepdetector.start_dp
        self.end_dp = left_stepdetector.end_dp
        self.timestamps = min([left_stepdetector.timestamps, right_stepdetector.timestamps], key=len)

    @staticmethod
    def identify_bouts_one(step_detector):
        freq = step_detector.freq
        steps = step_detector.step_indices
        timestamps = step_detector.timestamps[steps]
        step_lengths = step_detector.step_lengths
        steps_df = pd.DataFrame({'step_index': steps, 'timestamp': timestamps, 'step_length': step_lengths})
        steps_df = steps_df.sort_values(by=['step_index'], ignore_index=True)

        # assumes Hz are the same
        bout_dict = {'start': [], 'end': [], 'number_steps': [], 'start_timestamp': [], 'end_timestamp': []}
        start_step = steps_df.iloc[0]  # start of bout step
        curr_step = steps_df.iloc[0]
        step_count = 1
        next_steps = None

        while curr_step is not None:
            # Assumes steps are not empty and finds the next step after the current step
            termination_bout_window = pd.Timedelta(15, unit='sec') if next_steps is None else pd.Timedelta(10, unit='sec')
            next_steps = steps_df.loc[(steps_df['timestamp'] <= termination_bout_window + curr_step['timestamp'])
                                      & (steps_df['timestamp'] > curr_step['timestamp'])]

            if not next_steps.empty:
                curr_step = next_steps.iloc[0]
                step_count += 1
            else:
                # stores bout
                if step_count >= 2:
                    start_ind = start_step['step_index']
                    end_ind = curr_step['step_index'] + curr_step['step_length']
                    bout_dict['start'].append(start_ind)
                    bout_dict['end'].append(end_ind)
                    bout_dict['number_steps'].append(step_count)
                    bout_dict['start_timestamp'].append(start_step['timestamp'])
                    bout_dict['end_timestamp'].append(curr_step['timestamp'] + pd.Timedelta(curr_step['step_length'] / freq, unit='sec'))

                # resets state and creates new bout
                step_count = 1
                next_curr_steps = steps_df.loc[steps_df['timestamp'] > curr_step['timestamp']]
                curr_step = next_curr_steps.iloc[0] if not next_curr_steps.empty else None
                start_step = curr_step
                next_steps = None

        bout_num_df = pd.DataFrame(bout_dict)
        return bout_num_df

    @staticmethod
    def find_overlapping_times(left_bouts, right_bouts):
        # merge based on step index
        export_dict = {'start':[], 'end':[], 'number_steps': [], 'start_timestamp': [], 'end_timestamp':[]}
        all_bouts = pd.concat([left_bouts, right_bouts])
        all_bouts = all_bouts.sort_values(by=['start_timestamp'], ignore_index=True)
        all_bouts['overlaps'] = (all_bouts['start_timestamp'] < all_bouts['end_timestamp'].shift()) & (all_bouts['start_timestamp'].shift() < all_bouts['end_timestamp'])
        all_bouts['intersect_id'] = (((all_bouts['overlaps'].shift(-1) == True) & (all_bouts['overlaps'] == False)) |
                                      ((all_bouts['overlaps'].shift() == True) & (all_bouts['overlaps'] == False))).cumsum()
        for intersect_id, intersect in all_bouts.groupby('intersect_id'):
            # if there are no overlaps i want to iterate each individual bout
            if not intersect['overlaps'].any():
                for i, row in intersect.iterrows():
                    export_dict['start'].append(row['start'])
                    export_dict['end'].append(row['end'])
                    export_dict['number_steps'].append(row['number_steps'])
                    export_dict['start_timestamp'].append(row['start_timestamp'])
                    export_dict['end_timestamp'].append(row['end_timestamp'])
            else:
                export_dict['start'].append(intersect['start'].min())
                export_dict['end'].append(intersect['end'].max())
                export_dict['number_steps'].append(intersect['number_steps'].sum())
                export_dict['start_timestamp'].append(intersect['start_timestamp'].min())
                export_dict['end_timestamp'].append(intersect['end_timestamp'].max())

        df = pd.DataFrame(export_dict)
        df = df.sort_values(by=['start_timestamp'], ignore_index=True)
        df['overlaps'] = (df['start_timestamp'] < df['end_timestamp'].shift()) & (df['start_timestamp'].shift() < df['end_timestamp'])

        # if there are no overlaps
        if not df['overlaps'].any():
            df = df.drop(['overlaps'], axis=1)
            return df
        else:
            return WalkingBouts.find_overlapping_times(df, pd.DataFrame())

    @staticmethod
    def identify_bouts(left_stepdetector, right_stepdetector):
        """
        Identifies the bouts within the left and right acceleromter datas.
        The algorithm finds bouts that have 3 bilateral steps within a 15 second window
        """
        left_step_i = left_stepdetector.step_indices
        right_step_i = right_stepdetector.step_indices
        assert left_stepdetector.freq == right_stepdetector.freq
        freq = left_stepdetector.freq

        # merge into one list
        steps = np.concatenate([left_step_i, right_step_i])
        step_lengths = np.concatenate(left_stepdetector.step_lengths[left_step_i], right_stepdetector.step_lengths[right_step_i])
        foot = np.concatenate([['L'] * len(left_step_i), ['R'] * len(right_step_i)])
        timestamps = np.concatenate([left_stepdetector.timestamps[left_step_i], right_stepdetector.timestamps[right_step_i]])
        steps_df = pd.DataFrame({'step_index': steps, 'foot': foot, 'timestamp': timestamps, 'step_length': step_lengths})
        steps_df = steps_df.sort_values(by=['step_index'], ignore_index=True)

        # assumes Hz are the same
        bout_dict = {'start': [], 'end': [], 'bilateral_steps': [], 'start_timestamp': [], 'end_timestamp': []}
        start_step = steps_df.iloc[0]  # start of bout step
        curr_step = steps_df.iloc[0]
        bilateral_count = 0
        next_steps = None

        if steps_df.empty:
            return pd.DataFrame(bout_dict)

        while curr_step is not None:
            # Assumes steps are not empty
            termination_bout_window = pd.Timedelta(15, unit='sec') if next_steps is None else pd.Timedelta(10, unit='sec')
            next_steps = steps_df.loc[(steps_df['foot'] != curr_step['foot'])
                                      & (steps_df['timestamp'] <= termination_bout_window + curr_step['timestamp'])
                                      & (steps_df['timestamp'] > curr_step['timestamp'])]

            if not next_steps.empty:
                # iterate to next step
                curr_step = next_steps.iloc[0]
                bilateral_count += 1 if curr_step['foot'] != start_step['foot'] else 0
            else:
                # store/reset variables. begin new bout
                if bilateral_count >= 3:
                    start_ind = start_step['step_index']
                    end_ind = curr_step['step_index'] + curr_step['step_length']
                    bout_dict['start'].append(start_ind)
                    bout_dict['end'].append(end_ind)
                    bout_dict['bilateral_steps'].append(bilateral_count)
                    bout_dict['start_timestamp'].append(start_step['timestamp'])
                    bout_dict['end_timestamp'].append(curr_step['timestamp'] + pd.Timedelta(curr_step['step_length'] / freq, unit='sec'))

                bilateral_count = 0
                next_curr_steps = steps_df.loc[steps_df['timestamp'] > curr_step['timestamp']]
                curr_step = next_curr_steps.iloc[0] if not next_curr_steps.empty else None
                start_step = curr_step
                next_steps = None

        bout_num_df = pd.DataFrame(bout_dict)
        bout_num_df['left_cycle_count'] = [len(steps_df.loc[(steps_df['foot'] == 'L')
                                                            & (steps_df['step_index'] >= bout_num_df.iloc[i]['start'])
                                                            & (steps_df['step_index'] <= bout_num_df.iloc[i]['end'])]) for i in bout_num_df.index]
        bout_num_df['right_cycle_count'] = [len(steps_df.loc[(steps_df['foot'] == 'R')
                                                             & (steps_df['step_index'] >= bout_num_df.iloc[i]['start'])
                                                             & (steps_df['step_index'] <= bout_num_df.iloc[i]['end'])]) for i in bout_num_df.index]

        return bout_num_df

    def export_split_bouts(self, output_dir='figures'):
        """
        Takes in output dir and outputs images of the bouts into the output dir as .png images
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, row in self.bout_num_df.iterrows():
            fig = self.plot_single_bout(i, True)
            fig.savefig(os.path.join(output_dir, 'plot%d.png' % (i)))
            fig.clf()

    def plot_single_bout(self, bout_num, return_plt=False):
        """
        Shows single bout or start and end time range.
        Input is either the bout_num
        """
        row = self.bout_num_df.iloc[bout_num]
        pad_start = max(row['start'] - self.freq, 0)
        pad_end = min(row['end'] + self.freq, self.sig_length)
        fig = self.plot(pad_start, pad_end, return_plt)
        if not return_plt:
            fig.show()
        else:
            return fig

    def export_steps(self):
        bout_steps = []
        for i, row in self.bout_num_df.iterrows():
            start = row['start_timestamp'] - pd.Timedelta(1, unit='sec')
            end = row['end_timestamp'] + pd.Timedelta(1, unit='sec')

            left_bout_step_df = self.left_step_df.loc[(self.left_step_df['step_time'] > start) & (self.left_step_df['step_time'] < end)]
            right_bout_step_df = self.right_step_df.loc[(self.right_step_df['step_time'] > start) & (self.right_step_df['step_time'] < end)]

            bout_step_df = pd.concat([left_bout_step_df, right_bout_step_df])
            bout_step_df['gait_bout_num'] = i
            bout_steps.append(bout_step_df)

        bout_step_summary = pd.concat(bout_steps)
        bout_step_summary.sort_values(by=['gait_bout_num', 'foot', 'step_time'])
        bout_step_summary.reset_index(drop=True)

        return bout_step_summary

    def plot_accel(self, start=-1, end=-1, return_plt=False, margin=3):
        if not (start > -1 and end > -1):
            start = 0
            end = self.sig_length

        margin_dp = margin * self.freq
        duration = end - start
        start = start - margin_dp if start - margin_dp > 0 else 0
        end = end + margin_dp if end + margin_dp < self.sig_length else self.sig_length - 1

        if start == 0 or end == self.sig_length:
            margin = 0

        start_time = start / self.freq
        time_range = np.linspace(-margin, duration/self.freq+margin, num=end-start)

        bouts = np.zeros(self.sig_length)
        for i, row in self.bout_num_df.iterrows():
            bouts[row['start']:row['end']] = 5
        states_legend = ['stance', 'pushoff', 'swing down', 'swing up', 'heel strike']

        fig, ax = plt.subplots(1,1, sharex=True, figsize=(15,8))

        ax.set_title('Left/Right Accelerometer Data')
        ax.plot(time_range, self.left_data[start:end], 'r-', label='Left Accel')
        ax.plot(time_range, self.right_data[start:end], 'b-', label='Right Accel')
        # ax.axvline(time_range[0] + margin)
        # ax.axvline(time_range[-1] - margin)
        ax.set_xticks(np.arange(min(time_range), max(time_range)+1, 1), minor=True)
        ax.legend(loc='upper left')
        ax.grid(which='both')

        fig.tight_layout()
        if not return_plt:
            fig.show()
        else:
            return fig

    def plot(self, start=-1, end=-1, return_plt=False, include_bouts=True, margin=3):
        """
        Plots the overlayed accelerometer data, left and right states in accel data, and bouts detected
        ***margin is in seconds
        """
        if not (start > -1 and end > -1):
            start = 0
            end = self.sig_length

        margin_dp = margin * self.freq
        duration = end - start
        start = start - margin_dp if start - margin_dp > 0 else 0
        end = end + margin_dp if end + margin_dp < self.sig_length else self.sig_length - 1

        if start == 0 or end == self.sig_length:
            margin = 0

        start_time = start / self.freq
        time_range = np.linspace(-margin, duration/self.freq+margin, num=end-start)

        bouts = np.zeros(self.sig_length)
        for i, row in self.bout_num_df.iterrows():
            bouts[int(row['start']):int(row['end'])] = 5
        states_legend = ['stance', 'pushoff', 'swing down', 'swing up', 'heel strike']

        fig, axs = plt.subplots(3,1, sharex=True, figsize=(15,8))

        axs[0].set_title('Left/Right Accelerometer Data')
        axs[0].plot(time_range, self.left_data[start:end], 'r-', label='Left Accel')
        axs[0].plot(time_range, self.right_data[start:end], 'b-', label='Right Accel')
        axs[0].axvline(time_range[0] + margin)
        axs[0].axvline(time_range[-1] - margin)
        axs[0].set_xticks(np.arange(min(time_range), max(time_range)+1, 1), minor=True)
        axs[0].grid(which='both')

        axs[1].plot(time_range, self.left_states[start:end], "r-", label='Left Accel')
        axs[1].plot(time_range, self.right_states[start:end], "b-", label='Right Accel')
        axs[1].axvline(time_range[0] + margin)
        axs[1].axvline(time_range[-1] - margin)
        axs[1].set_xticks(np.arange(min(time_range), max(time_range)+1, 1), minor=True)
        axs[1].fill_between(time_range, 0, bouts[start:end], alpha=0.5)
        axs[1].set_title('States of Steps in Accelerometer Data')
        axs[1].set_yticks(np.arange(len(states_legend)))
        axs[1].set_yticklabels(states_legend)
        axs[1].legend(loc='upper left')
        axs[1].grid(which='both')

        if include_bouts:
            axs[2].plot(time_range, bouts[start:end], "g-")
            axs[2].set_title('Bouts Detected')
            axs[2].set_xlabel('Time (s)')
            axs[2].grid(which='both')
        else:
            filtered_legend = ['', 'swing down', 'swing up', 'heel strike',
                               'next pushoff too close', 'pushoff mean too far', 'mid_swing_peak']
            axs[2].plot(time_range, self.left_steps_failed[start:end], "ro")
            axs[2].plot(time_range, self.right_steps_failed[start:end], "bo")
            axs[2].set_xticks(np.arange(min(time_range), max(time_range)+1, 1), minor=True)
            axs[2].set_yticks(np.arange(len(filtered_legend)))
            axs[2].set_yticklabels(filtered_legend)
            axs[2].set_title('Failed Steps')
            axs[2].set_xlabel('Time (s)')
            axs[2].grid(which='both')

        fig.tight_layout()
        axs[2].axvline(time_range[0] + margin)
        axs[2].axvline(time_range[-1] - margin)

        if not return_plt:
            fig.show()
        else:
            return fig

    def export_bouts(self, name='UNKNOWN'):
        summary = pd.DataFrame({
            'name': name,
            'gait_bout_num': self.bout_num_df.index,
            'start_timestamp': self.bout_num_df['start_timestamp'],
            'end_timestamp': self.bout_num_df['end_timestamp'],
            'start_dp': self.bout_num_df['start'],
            'end_dp': self.bout_num_df['end'],
            'bout_length_dp': self.bout_num_df['end'] - self.bout_num_df['start'],
            'bout_length_sec': [(row['end_timestamp'] - row['start_timestamp']).total_seconds() for i, row in self.bout_num_df.iterrows()]
        })
        if self.left_states.shape == self.right_states.shape:
            states = self.left_states + self.right_states
            summary['gait_time_sec'] = [len(np.where(states[row['start_dp']:row['end_dp']] > 0)[0]) / self.freq for i, row in summary.iterrows()]
        return summary
    
if __name__ == '__main__':
    start_time = '2019-08-22 10:05:16'
    duration = 1000
    l_file = r'/Users/matthewwong/Documents/coding/nimbal/data/OND06_SBH_2891_GNAC_ACCELEROMETER_LAnkle.edf'
    r_file = r'/Users/matthewwong/Documents/coding/nimbal/data/OND06_SBH_2891_GNAC_ACCELEROMETER_RAnkle.edf'
    wb = WalkingBouts(l_file, r_file, start_time=start_time, duration_sec=duration)
