#!/usr/bin/python3

"""
Created on Wed Oct 20 20:40:34 2020

@author: Alex Kong
"""

import datetime
from decimal import Decimal
import gzip
import json
import logging
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pathlib
import os
import random
import re
import requests
import socket
import subprocess
from subprocess import Popen, PIPE
import sys
import threading
import time
import urllib.request


INVALID_NUMBER = -99999.99

def standardize_hkex_data():

    parent_dir = pathlib.Path(__file__).parent.absolute()    

    df = pd.read_csv(os.path.join(parent_dir, 'hkex_raw.csv'))

    indicators = ['stock_code',
              'company_name',
              'annual_gross_margin', 
              'asset_turnover_annual',
              'cash_share_annual', # cash per share 
              'current_ratio_annual', 
              'eps_basic_excl_extra_annual', # Earnings per share excluding extraordinary items
              'fcf_share_ttm', # free cash flow per share Trailing 12 months 
              'inventory_turnover_annual',
              'lt_debt_equity_annual', # long-term debt to equity
              'net_interest_coverage_annual', # Net interest coverage ratio
              'net_profit_margin_annual', 
              'operating_margin_annual', 
              'quick_ratio_annual', 
              'receivables_turnover_annual',
              'roe_ttm', # return on equity, trailing 12 months
              'roi_annual',  # return on investment
              'roa_rfy', #  return on assets
              'tangible_book_annual',
              'total_debt_cagr_5y', 
              'total_debt_equity_annual']
    
    df['invalid_rate'] = 0
 #   print(df[indicators[3]])
 #   return
    for i in range(2, len(indicators)):

    #    print(i)
        valid_mean = df[indicators[i]].loc[df[indicators[i]] > INVALID_NUMBER].mean()
        valid_std = df[indicators[i]].loc[df[indicators[i]] > INVALID_NUMBER].std()
        invalid_count = len(df[indicators[i]].loc[df[indicators[i]] <= INVALID_NUMBER])        
        df['invalid_rate'].loc[df[indicators[i]] <= INVALID_NUMBER] = df['invalid_rate'].loc[df[indicators[i]] <= INVALID_NUMBER] + 1

        df[indicators[i]].loc[df[indicators[i]] > INVALID_NUMBER] = (df[indicators[i]] - valid_mean) / valid_std
        df[indicators[i]].loc[df[indicators[i]] <= INVALID_NUMBER] = np.array([0] * invalid_count)

    print(df[['stock_code', 'invalid_rate']])
    df['invalid_rate'] = df['invalid_rate'] / (len(indicators) - 3)
    print(df[['stock_code', 'invalid_rate']])
    
    df.to_csv(os.path.join(parent_dir, 'hkex_standardized.csv'), index = False, header=True)
 
if __name__ == '__main__':    
    
    standardize_hkex_data()

