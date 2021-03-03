#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:43:53 2021

@author: matthewwong
"""

from macro_gait.WalkingBouts import WalkingBouts
import pandas as pd

def get_swing_stance_times():
    pass
    

if __name__ == '__main__':
    path1 = '/Users/matthewwong/Documents/coding/fydp/walking1.csv'
    path2 = '/Users/matthewwong/Documents/coding/fydp/walking2.csv'
    pushoff_df = pd.read_csv('/Users/matthewwong/Documents/coding/fydp/macro_gait/pushoff_OND07_left.csv')
    bouts = WalkingBouts(path1, path2, left_kwargs={'pushoff_df': pushoff_df}, right_kwargs={'pushoff_df': pushoff_df})
    df = bouts.export_steps()
    pass
