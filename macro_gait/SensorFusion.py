#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:49:54 2021

@author: matthewwong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from macro_gait.StepDetection import StepDetection

class SensorFusion(StepDetection):
    def __init__(self, path, **kwargs):
        super(SensorFusion, self).__init__(accel_path=path, **kwargs)
        self.cop_locs = {'p1': (3,2), 'p2': (5,2), 'p3': (1,11), 'p4': (2,17), 'p5': (5,19), 'p6': (8,19)}
        self.force_data = SensorFusion.get_force_data(path)
        self.IMU_data = SensorFusion.get_IMU_data(path)
        self.COP_data = SensorFusion.get_cop(path, self.cop_locs)
        # instead of max, use their weight
        self.norm_force_data = (self.force_data - self.force_data.min(axis=0)) / (self.force_data.max(axis=0) - self.force_data.min(axis=0)).copy() 

    @staticmethod
    def get_force_data(path):
        df = pd.read_csv(path)
        pressure_axes = ['p1','p2','p3','p4','p5','p6']
        data = df[pressure_axes]
        return data
    
    @staticmethod
    def get_IMU_data(path):
        df = pd.read_csv(path)
        axes = ['gx', 'gy', 'gz', 'ax', 'ay', 'az']
        data = df[axes]
        return data
    
    @staticmethod
    def get_cop(path, locs):
        df = SensorFusion.get_force_data(path)
        df['GRF'] = df.sum(axis=1)
        df['COP_ML'] = 0
        df['COP_AP'] = 0
        for k, v in locs.items():
            df['COP_ML'] += df[k] * v[0] / df['GRF']
            df['COP_AP'] += df[k] * v[1] / df['GRF']
        
        return df[['COP_ML', 'COP_AP']].copy()
        
        
    
    @staticmethod
    def hyst(x, th_lo, th_hi, initial = False):
        hi = x >= th_hi
        lo_or_hi = (x <= th_lo) | hi
        ind = np.nonzero(lo_or_hi)[0]
        if not ind.size: # prevent index error if ind is empty
            return np.zeros_like(x, dtype=bool) | initial
        cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
        return np.where(cnt, hi[ind[cnt-1]], initial)
    
    def export_steps(self):
        steps = super().export_steps()
        force_df = self.norm_force_data.copy()
        for col in force_df:
            force_df[col] = ~self.hyst(force_df[col].to_numpy(), 0.5, 0.65)
        newton_force_df = self.force_data.copy()
        newton_force_df['GRF'] = force_df.sum(axis=1)
        newton_force_df['timestamps'] = self.timestamps
        force_df['timestamps'] = self.timestamps
        for i, step in steps.iterrows():
            ms_ind = self.mid_swing_peak_detect(step['step_index'] + self.pushoff_len)
                
            # basically checks if foot is off the ground up until midswing
            # TODO: we want to check more of the force sensors!
            df = force_df.loc[(force_df['timestamps'] > step['step_time'] + pd.Timedelta(self.pushoff_time, unit='s')) & (force_df['timestamps'] < self.timestamps[ms_ind])]
            if not df['p1'].all():
                steps.loc[i, 'step_state'] = 'heel_on_ground_during_swing'
                continue
            
            # TODO: base it on force sensor
            po_list = np.where((force_df['p1'] == 0) & (force_df['p1'].shift(-1) == 1))[0]
            po_ind = min(po_list, key=lambda x:abs(x-step['step_index']))
                         
            # checks if correction of pushoff is too far
            if abs(po_ind - step['step_index']) > 30:
                steps.loc[i, 'step_state'] = 'pushoff_too_far_from_pressure'
                continue
            
            ss_ind = step['step_index'] + self.pushoff_len
            hs_ind = np.where(self.heel_strike_detect(ms_ind) < self.heel_strike_threshold)[0][0] + ms_ind
            fd_ind = int(hs_ind + self.foot_down_time * self.freq)
            
            step_length_sec =  (fd_ind - ss_ind) / self.freq
            avg_speed = np.trapz(self.xz_data[po_ind:fd_ind], dx=(1/self.freq))
            avg_speed = np.trapz(self.xz_data[ss_ind:fd_ind], dx=(1/self.freq))
            swing_speed = np.trapz(self.xz_data[po_ind:hs_ind], dx=(1/self.freq))
            
            # temporal / spatial stats
            steps.loc[i, 'step_time'] = self.timestamps[po_ind]
            steps.loc[i, 'step_len_sec'] = (fd_ind - po_ind) / self.freq
            steps.loc[i, 'step_index'] = po_ind
            steps.loc[i, 'step_state'] = 'success'
            steps.loc[i, 'swing_start_time'] = self.timestamps[ss_ind]
            steps.loc[i, 'mid_swing_time'] =  self.timestamps[ms_ind] 
            steps.loc[i, 'heel_strike_time'] = self.timestamps[hs_ind]
            steps.loc[i, 'foot_down_time'] = self.timestamps[fd_ind]
            steps.loc[i, 'swing_start_accel'] = self.data[ss_ind]
            steps.loc[i, 'mid_swing_accel'] = self.data[ms_ind]
            steps.loc[i, 'heel_strike_accel'] = self.data[hs_ind]
            steps.loc[i, 'step_length_sec'] = step_length_sec
            steps.loc[i, 'max_angular_vel'] = self.IMU_data.loc[po_ind:fd_ind, ['gx', 'gy', 'gz']].abs().values.max()
            steps.loc[i, 'avg_speed'] = avg_speed
            steps.loc[i, 'swing_speed'] = swing_speed
            
            # force and pressure stats
            step_force_df = newton_force_df.loc[(newton_force_df['timestamps'] > self.timestamps[po_ind]) & (newton_force_df['timestamps'] < self.timestamps[fd_ind])].copy()
            step_force_df['heel_pressure'] = step_force_df['p1'] + step_force_df['p2']
            step_force_df['forefoot_pressure'] = step_force_df['p4'] + step_force_df['p5'] + step_force_df['p6']
            
            heeltime = step_force_df.loc[step_force_df['heel_pressure'] < step_force_df['heel_pressure'].max() * 0.8, 'heel_pressure'].shape[0] * 1/self.freq
            # TODO: get rid of 0.1
            forefoot_ratio = step_force_df['forefoot_pressure'].iloc[step_force_df['heel_pressure'].argmax()] / (step_force_df['forefoot_pressure'].max() + 0.1)
            forefoot_heel_ratio = step_force_df['forefoot_pressure'].max() / step_force_df['heel_pressure'].max()
            
            steps.loc[i, 'heeltime'] = heeltime
            steps.loc[i, 'max_forefoot'] = step_force_df['forefoot_pressure'].max()
            steps.loc[i, 'forefoot_ratio'] = forefoot_ratio
            steps.loc[i, 'forefoot_heel_ratio'] = forefoot_heel_ratio
            steps.loc[i, 'GRF'] = step_force_df['GRF'].mean()
            steps.loc[i, 'AP_len'] = self.COP_data['COP_AP'].max() - self.COP_data['COP_AP'].min()
            steps.loc[i, 'ML_len'] = self.COP_data['COP_ML'].max() - self.COP_data['COP_ML'].min()
            steps.loc[i, 'AP_pos'] = self.COP_data['COP_AP'].mean()
            steps.loc[i, 'ML_pos'] = self.COP_data['COP_ML'].mean()
                    
        return steps
    
    def plot(self):
        dp_range = np.arange(self.start_dp, self.end_dp)
        ax1 = plt.subplot(311)
        ax1.set_title('Accelerometer Data')
        plt.plot(dp_range, self.data, 'r-')
        plt.grid(True)
        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_title('Force Data Raw')
        plt.plot(dp_range, self.force_data['p1'])
        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_title('Force Data')
        plt.plot(dp_range, self.hyst(self.norm_force_data['p1'].to_numpy(), 0.5, 0.65))
        plt.show()
    
    def plot_accel_mean(self, show_plt=False):
        steps = self.export_steps()
        
        step_len_ind = int((steps['step_len_sec'] * self.freq).mean())
        step_sig_list = []
        
        mean_po = (steps['swing_start_time'] - steps['step_time']).dt.total_seconds().mean()
        mean_sd = (steps['mid_swing_time'] - steps['swing_start_time']).dt.total_seconds().mean()
        mean_su = (steps['heel_strike_time'] - steps['mid_swing_time']).dt.total_seconds().mean()
        mean_fd = (steps['foot_down_time'] - steps['heel_strike_time']).dt.total_seconds().mean()
        for i, step in steps.iterrows():
            start_ind = step['step_index']
            end_ind = start_ind + step_len_ind
            step_sig = self.data[start_ind:end_ind]
            step_sig_list.append(step_sig)

        mean_sig = np.mean(step_sig_list, axis=0)
        std_sig = np.std(step_sig_list, axis=0)
        max_sig = np.max(step_sig_list, axis=0)
        min_sig = np.min(step_sig_list, axis=0)
        
        time_range = np.linspace(0,steps['step_len_sec'].mean(),len(mean_sig))
        alpha = 0.4
        
        fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle('Mean Acceleration in Step')
        ax[0].plot(time_range, mean_sig, label='mean')
        ax[0].plot(time_range, mean_sig+std_sig, 'r--', label='std', alpha=alpha)
        ax[0].plot(time_range, mean_sig-std_sig, 'r--', alpha=alpha)
        ax[0].plot(time_range, max_sig, 'g:', label='max/min', alpha=alpha)
        ax[0].plot(time_range, min_sig, 'g:', alpha=alpha)
        ax[0].set_ylabel('acceleration (m/s^2)')
        ax[0].legend()
        
        ax[1].barh([''], mean_po, label='pushoff', height=0.5)
        ax[1].barh([''], mean_sd, left=mean_po, label='swing down', height=0.5)
        ax[1].barh([''], mean_su, left=(mean_sd+mean_po), label='swing up', height=0.5)
        ax[1].barh([''], mean_fd, left=(mean_po + mean_sd + mean_su),label='foot down', height=0.5)
        ax[1].set_xlabel('time (s)')
        ax[1].set_ylabel('step phases')
        ax[1].legend()
        ax[1].set_yticks('hello')
    
        
        if show_plt:
            fig.show()
        return fig
    
    def plot_force_mean(self, show_plt=False):
        steps = self.export_steps()
        GRF = self.force_data.sum(axis=1).to_numpy()
        step_len_ind = int((steps['step_len_sec'] * self.freq).mean())
        step_sig_list = []
        
        for i, step in steps.iterrows():
            start_ind = step['step_index']
            end_ind = start_ind + step_len_ind
            step_sig = GRF[start_ind:end_ind]
            step_sig_list.append(step_sig)

        mean_sig = np.mean(step_sig_list, axis=0)
        std_sig = np.std(step_sig_list, axis=0)
        max_sig = np.max(step_sig_list, axis=0)
        min_sig = np.min(step_sig_list, axis=0)
        
        time_range = np.linspace(0,steps['step_len_sec'].mean(),len(mean_sig))
        alpha = 0.4
        plt.clf()
        plt.plot(time_range, mean_sig, label='mean')
        plt.plot(time_range, mean_sig+std_sig, 'r--', label='std', alpha=alpha)
        plt.plot(time_range, mean_sig-std_sig, 'r--', alpha=alpha)
        plt.plot(time_range, max_sig, 'g:', label='max/min', alpha=alpha)
        plt.plot(time_range, min_sig, 'g:', alpha=alpha)
        plt.title('Mean Force in Step')
        plt.xlabel('time (s)')
        plt.ylabel('force (N)')
        plt.legend()
        
        if show_plt:
            plt.show()
        return plt
    
    def plot_cop_mean(self, show_plt=False, mirror=False):
        steps = self.export_steps()
        
        COP_x = self.COP_data['COP_ML']
        COP_y = self.COP_data['COP_AP']
        GRF = self.force_data.sum(axis=1).to_numpy()
        
        step_len_ind = int((steps['step_len_sec'] * self.freq).mean())
        x_sig_list = []
        y_sig_list = []
        step_sig_list = []
        
        for i, step in steps.iterrows():
            start_ind = step['step_index']
            end_ind = start_ind + step_len_ind
            step_sig = GRF[start_ind:end_ind]
            
            x_sig_list.append(self.COP_data.loc[start_ind:end_ind, 'COP_ML'])
            y_sig_list.append(self.COP_data.loc[start_ind:end_ind, 'COP_AP'])
            step_sig_list.append(step_sig)

        # TODO: update to real COP
        x = -np.mean(x_sig_list, axis=0) if mirror else np.mean(x_sig_list, axis=0)
        y = np.mean(y_sig_list, axis=0)
        
        #fig, ax = plt.subplots()
        plt.clf()
        plt.title('Center of Pressure in Step')
        max_x, max_y = np.max(list(self.cop_locs.values()), axis=0)
        
        #plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
        plt.ylim([0, max_y + 5])
        plt.xlim([-max_x-1, 0]) if mirror else plt.xlim([0, max_x + 1])
        for k,v in self.cop_locs.items():
            v = (-v[0], v[1]) if mirror else v 
            plt.text(v[0], v[1], k, ha="center", va="center",
             bbox = dict(boxstyle=f"circle,pad={1}", fc="lightgrey", alpha=0.3))
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        
        if show_plt:
            plt.show()
        return plt
    
    def get_mean_sigs(self):
        steps = self.export_steps()
        GRF = self.force_data.sum(axis=1).to_numpy()
        step_len_ind = int((steps['step_len_sec'] * self.freq).mean())
        cop_x_list = []
        cop_y_list = []
        force_list = []
        accel_list = []
        
        for i, step in steps.iterrows():
            start_ind = step['step_index']
            end_ind = start_ind + step_len_ind
            cop_x_list.append(self.COP_data.loc[start_ind:end_ind, 'COP_ML'])
            cop_y_list.append(self.COP_data.loc[start_ind:end_ind, 'COP_AP'])
            force_list.append(GRF[start_ind:end_ind])
            accel_list.append(self.data[start_ind:end_ind])
            
        cop_list = np.sqrt(np.mean(cop_x_list, axis=0)**2 + np.mean(cop_x_list,axis=0)**2)
        
        return cop_list, np.mean(force_list, axis=0), np.mean(accel_list, axis=0)
        
        
        
        
        
    

if __name__ == "__main__":
    path = '/Users/matthewwong/Documents/coding/fydp/walking2.csv'
    pushoff_df = '/Users/matthewwong/Documents/coding/fydp/macro_gait/pushoff_OND07_left.csv'
    f = SensorFusion(path, pushoff_df=pd.read_csv(pushoff_df))
    f.plot_cop_mean(show_plt=True, mirror=True)
    