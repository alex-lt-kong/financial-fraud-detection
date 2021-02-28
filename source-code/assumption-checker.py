#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:40:34 2020

@author: Alex Kong
"""


import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os

lib = importlib.import_module('data-analysis-library')
parent_dir = pathlib.Path(__file__).parent.absolute()

def normality_test(df):
    
    for column in df.columns[2:]:
        print(f'\n\n ===== {column} =====')
        lib.lec3_shapiro_wilk_normality_test(dataset = np.array(df[column]),
                                     significance_level = 0.05)
        lib.lec3_kolmogorov_smirnov_normality_test(dataset = np.array(df[column]),
                                        dataset_standardized = True,
                                        significance_level = 0.05)
        lib.lec3_anderson_darling_normality_test(dataset = np.array(df[column]))
        
        plt.hist(df[column], bins = 250)
        plt.show()

def multicollinearity(df):
    
    df.drop(labels = 'stock_code', axis = 1, inplace = True)
    df.drop(labels = 'company_name', axis = 1, inplace = True)

    lib.__lec6_multicollinearity_detector(
            dataset_with_all_independent_variables = df,
            intercept_included = True,
            verbosity = 2)
    
    print('\n\n\n')
    df.drop(labels = 'net_profit_margin_annual', axis = 1, inplace = True)
    df.drop(labels = 'roa_rfy', axis = 1, inplace = True)
     
    lib.__lec6_multicollinearity_detector(
            dataset_with_all_independent_variables = df,
            intercept_included = True,
            verbosity = 2)

def main():

    df = pd.read_csv('hkex_standardized.csv')
    
    normality_test(df)
    multicollinearity(df)

    
if __name__ == '__main__':
    main()