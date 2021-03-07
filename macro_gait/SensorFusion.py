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
        newton_force_df = self.force_data
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
    

if __name__ == "__main__":
    path = '/Users/matthewwong/Documents/coding/fydp/walking2.csv'
    pushoff_df = '/Users/matthewwong/Documents/coding/fydp/macro_gait/pushoff_OND07_left.csv'
    f = SensorFusion(path, pushoff_df=pd.read_csv(pushoff_df))
    steps = f.export_steps()
    