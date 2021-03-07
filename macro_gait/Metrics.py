#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:43:53 2021

@author: matthewwong
"""

from macro_gait.WalkingBouts import WalkingBouts
import pandas as pd
import numpy as np

def get_general_stats(steps):
    l_steps, r_steps = steps.loc[(steps['foot'] == 'left') & (steps['step_state'] == 'success')], steps.loc[(steps['foot'] == 'right') & (steps['step_state'] == 'success')]
    l_total = (l_steps['foot_down_time'] - l_steps['step_time']).dt.total_seconds()
    r_total = (r_steps['foot_down_time'] - r_steps['step_time']).dt.total_seconds()
    l_cadence, r_cadence = [], []
    
    
    l_speed = l_steps['avg_speed']
    r_speed = r_steps['avg_speed']
    l_length = l_speed * l_total
    r_length = r_speed * r_total
    
    
    for i, step in l_steps.groupby('gait_bout_num'):
        bout_time = (step['foot_down_time'].max() - step['step_time'].min()).total_seconds()
        l_cadence.append(step.shape[0]/(bout_time/60))
        
    for i, step in r_steps.groupby('gait_bout_num'):
        bout_time = (step['foot_down_time'].max() - step['step_time'].min()).total_seconds()
        r_cadence.append(step.shape[0]/(bout_time/60))
            
    stats = [l_total, r_total,
             l_cadence, r_cadence,
             l_length, r_length,
             l_speed, r_speed]
    
    general_df = pd.DataFrame(
        {'mean': [np.nanmean(x) for x in stats],
         'std': [np.nanstd(x) for x in stats],
         'median': [np.nanmedian(x) for x in stats],
         'min': [np.nanmin(x) for x in stats],
         'max': [np.nanmax(x) for x in stats]},
        index=[
            ['cycle_duration', 'cycle_duration', 'cadence', 'cadence', 'stride_length', 'stride_length', 'stride_velocity', 'stride_velocity'], 
            ['L', 'R', 'L', 'R', 'L', 'R', 'L', 'R']])
    general_df = get_asymmetry_stats(general_df)
    return general_df

def get_temporal_stats(steps):
    l_steps, r_steps = steps.loc[(steps['foot'] == 'left') & (steps['step_state'] == 'success')], steps.loc[(steps['foot'] == 'right') & (steps['step_state'] == 'success')]
    
    l_swings = (l_steps['heel_strike_time'] - l_steps['swing_start_time']).dt.total_seconds()
    l_loading = (l_steps['foot_down_time'] - l_steps['heel_strike_time']).dt.total_seconds()
    l_pushing = (l_steps['swing_start_time'] - l_steps['step_time']).dt.total_seconds()
    
    l_footdown = np.concatenate([(step['step_time'].shift(-1) - step['foot_down_time']).dt.total_seconds() for i, step in l_steps.groupby('gait_bout_num')])
    l_footdown[np.isnan(l_footdown)] = 0
    l_total = (l_steps['foot_down_time'] - l_steps['step_time']).dt.total_seconds() + l_footdown
    l_stances = l_total - l_swings
    
    r_swings = (r_steps['heel_strike_time'] - r_steps['swing_start_time']).dt.total_seconds()
    r_loading = (r_steps['foot_down_time'] - r_steps['heel_strike_time']).dt.total_seconds()
    r_pushing = (r_steps['swing_start_time'] - r_steps['step_time']).dt.total_seconds()
    r_footdown = np.concatenate([(step['step_time'].shift(-1) - step['foot_down_time']).dt.total_seconds() for i, step in r_steps.groupby('gait_bout_num')])
    r_footdown[np.isnan(r_footdown)] = 0
    r_total = (r_steps['foot_down_time'] - r_steps['step_time']).dt.total_seconds() + r_footdown
    r_stances = r_total - r_swings
    
    l_doublesupp = l_total - l_swings - r_swings
    r_doublesupp = r_total - l_swings - r_swings    
    # MEAN STATS
    # TODO: Add that stats foot-flat stats
    stats = [(l_stances/(l_swings+l_stances)), (r_stances/(r_swings+r_stances)),
            (l_swings/(l_swings+l_stances)), (r_swings/(r_swings+r_stances)),
            (l_loading/l_stances), (r_loading/r_stances),
            (l_pushing/l_stances), (r_pushing/r_stances),
            (l_footdown/l_stances), (r_footdown/r_stances),
            l_doublesupp, r_doublesupp]

    temporal_df = pd.DataFrame(
        {'mean': [np.nanmean(x) for x in stats],
         'std': [np.nanstd(x) for x in stats],
         'median': [np.nanmedian(x) for x in stats],
         'min': [np.nanmin(x) for x in stats],
         'max': [np.nanmax(x) for x in stats]},
        index=[
            ['stance', 'stance', 'swing', 'swing', 'loading', 'loading', 'pushing', 'pushing', 'foot-down', 'foot-down', 'double-support', 'double-support'], 
            ['L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R']])
    temporal_df = get_asymmetry_stats(temporal_df)
    return temporal_df

def get_spatial_stats(steps):
    l_steps, r_steps = steps.loc[(steps['foot'] == 'left') & (steps['step_state'] == 'success')], steps.loc[(steps['foot'] == 'right') & (steps['step_state'] == 'success')]
    l_ang_vel = l_steps['max_angular_vel']
    r_ang_vel = r_steps['max_angular_vel']
    l_swing_speed = l_steps['swing_speed']
    r_swing_speed = r_steps['swing_speed']
    
    stats = [l_ang_vel, r_ang_vel,
             l_swing_speed, r_swing_speed]
    
    spatial_df = pd.DataFrame(
        {'mean': [np.nanmean(x) for x in stats],
         'std': [np.nanstd(x) for x in stats],
         'median': [np.nanmedian(x) for x in stats],
         'min': [np.nanmin(x) for x in stats],
         'max': [np.nanmax(x) for x in stats]},
        index=[
            ['peak_ang_vel', 'peak_ang_vel', 'swing_speed', 'swing_speed'], 
            ['L', 'R', 'L', 'R']])
    
    spatial_df = get_asymmetry_stats(spatial_df)
    return spatial_df

def get_asymmetry_stats(df):
    concat_list = []
    
    for stat, temp_df in df.groupby(level=0):
        asymm = temp_df.xs(stat)['mean'].max() / temp_df.xs(stat)['mean'].min()
        temp_df['symm_index'] = asymm
        concat_list.append(temp_df)
    
    return pd.concat(concat_list).sort_index()
        
def get_pressure_stats(steps, bout_obj):
    l_steps, r_steps = steps.loc[(steps['foot'] == 'left') & (steps['step_state'] == 'success')], steps.loc[(steps['foot'] == 'right') & (steps['step_state'] == 'success')]    
            
    stats = [l_steps['GRF'], r_steps['GRF'], l_steps['heeltime'], r_steps['heeltime'],
             l_steps['forefoot_ratio'], r_steps['forefoot_ratio'], l_steps['max_forefoot'], r_steps['max_forefoot'],
             l_steps['forefoot_heel_ratio'], r_steps['forefoot_heel_ratio'],
             l_steps['AP_len'], r_steps['AP_len'], l_steps['ML_len'], r_steps['ML_len'],
             l_steps['AP_pos'], r_steps['AP_pos'], l_steps['ML_pos'], r_steps['ML_pos']]

    pressure_df = pd.DataFrame(
        {'mean': [np.nanmean(x) for x in stats],
         'std': [np.nanstd(x) for x in stats],
         'median': [np.nanmedian(x) for x in stats],
         'min': [np.nanmin(x) for x in stats],
         'max': [np.nanmax(x) for x in stats]},
        index=[
            ['GRF', 'GRF', 'heeltime', 'heeltime', 'max_forefoot', 'max_forefoot',
             'forefoot_ratio', 'forefoot_ratio' ,'forefoot_heel_ratio', 'forefoot_heel_ratio',
             'AP_len', 'AP_len', 'ML_len', 'ML_len', 'AP_pos', 'AP_pos', 'ML_pos', 'ML_pos'], 
            ['L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R']])
    pressure_df = get_asymmetry_stats(pressure_df)
    return pressure_df
        
            
                
        
if __name__ == '__main__':
    path1 = '/Users/matthewwong/Documents/coding/fydp/walking1.csv'
    path2 = '/Users/matthewwong/Documents/coding/fydp/walking2.csv'
    pushoff_df = pd.read_csv('/Users/matthewwong/Documents/coding/fydp/macro_gait/pushoff_OND07_left.csv')
    bouts_obj = WalkingBouts(path1, path2, left_kwargs={'pushoff_df': pushoff_df}, right_kwargs={'pushoff_df': pushoff_df})
    steps = bouts_obj.export_steps()
    bouts = bouts_obj.export_bouts()
    pp = get_temporal_stats(steps)
    gg = get_general_stats(steps)
    oo = get_spatial_stats(steps)
    ppp = get_pressure_stats(steps, bouts_obj)
