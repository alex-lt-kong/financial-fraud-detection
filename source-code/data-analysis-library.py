#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:36:26 2020

@author: Alex Kong
"""

from sklearn.linear_model import Ridge
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict
from typing import List
from typing import Union

import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms


def lec1_boxplot(dataset: pd.DataFrame, column: str, group_by = None, vertical = True, verbosity = 0):

    assert isinstance(dataset, pd.core.frame.DataFrame)

    dataset.boxplot(column = column, by = group_by, vert = vertical)
    plt.show()

    if verbosity >=1:
        print('Plot Interpretation:')
        print('(1) If sample is from a normal distribution, it is very unlikely to have outliners (o-shaped point). Suppose we can see a lot of outliners, we should suspect that the sample is not from a normal distribution.')

def lec1_lienchart(dataset_x: np.ndarray,
                  dataset_y: np.ndarray,
                  title = 'Untitled Chart',
                  x_title = 'Untitled independent variable',
                  y_title = 'Untitled dependent variable',
                  color = 'red',
                  linewidth = 0.5):

    assert len(dataset_x) == len(dataset_y)

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.plot(dataset_x, dataset_y, color = color, linewidth = linewidth)
    plt.show()

def lec1_qq_plot_aka_quantile_quantile_plot(dataset: pd.DataFrame,
                                           studentize_dataset: bool,
                                           title = 'Untitled Plot',
                                           verbosity = 0):

    #if np.isnan(dataset).any() and verbosity >= 1:
    #    print('Note: nan exists in the dataset. It appears to be fine but there is no harm in removing them.')

    if studentize_dataset:
        dataset = (dataset - dataset.mean()) / dataset.std()

    sm.qqplot(data=dataset, dist=sp.stats.distributions.norm, line='45')
    plt.title(title)
    plt.show()

def lec2_geometric_brownian_motion(mu_aka_risk_free_rate_or_drift = 0,
                                  sigma_aka_volatility = 0.3,
                                  share_price_at_t0 = 20,
                                  verbosity = 2):

    T = 2 # time period, the real interval is generated below as t
    μ = mu_aka_risk_free_rate_or_drift  # 0 means without drift.
    # This is very important: If the stock price is driven by the risk-neutral measure,
    # then it will have an expeccted value only discounted by the risk-free rate r.
    # For risk-neutral measure, mu is euqla to risk-free rate r.
    # However, for Merton Jump Diffusion Model (MDJ) Model, interest rate is not equal to mu
    σ = sigma_aka_volatility
    delta_t = 0.01 # step side
    S0 = share_price_at_t0
    N = round(T/delta_t)

    t = np.linspace(0, T, N)
    # np.linspace is not about vector space, it simply eturns evenly spaced numbers over a specified interval.

    dZ = np.random.standard_normal(size = N) * np.sqrt(delta_t)
    Z = np.cumsum(dZ) # the summation operation seen in the formula.

    X = (μ - 0.5 * σ ** 2) * t + σ * Z
    share_prices = np.append(S0, S0 * np.exp(X)) # This line will include S0 in the final result
    #share_prices = S0 * np.exp(X) # This line will NOT include S0 in the final result.
    # np.exp(x) just means e^x
    if verbosity > 0:
        plt.figure(7)
        plt.plot(share_prices)
        plt.title('Plot of share prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()
    # If you want to include S0 in share_prices, you cannot use plt.plot(t, share_prices)
    # since len(t) != len(share_prices)

    if verbosity > 1:
        lec1_qq_plot_aka_quantile_quantile_plot(dataset = (share_prices),
                                               studentize_dataset = True,
                                               title = 'QQ Plot of simulated returns',
                                               verbosity = 0)
        rt = np.log(share_prices)
        lec1_qq_plot_aka_quantile_quantile_plot(dataset = (rt),
                                               studentize_dataset = True,
                                               title = 'QQ Plot of simulated log returns',
                                               verbosity = 0)

    return share_prices

def lec2_monte_carlo_simulation_with_gbm(
                                  mu_aka_risk_free_rate_or_drift = 0,
                                  sigma_aka_volatility = 0.3,
                                  share_price_at_t0 = 20,
                                  number_of_simulations = 100,
                                  number_of_bins = 20,
                                  verbosity = 2):

    simulations = {}
    for i in range(number_of_simulations):
        simulations['simulation ' + str(i)] = lec2_geometric_brownian_motion(
                                  mu_aka_risk_free_rate_or_drift = mu_aka_risk_free_rate_or_drift,
                                  sigma_aka_volatility = sigma_aka_volatility,
                                  share_price_at_t0 = share_price_at_t0,
                                  verbosity = 0)

    simulations = pd.DataFrame(simulations)
    if verbosity > 0:
        simulations.plot(figsize=(10, 7), grid = True, legend = False)
        plt.title('Plot of {} simulations'.format(number_of_simulations))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()

    if verbosity > 1:
        plt.hist(x=simulations.iloc[-1], bins=number_of_bins, histtype='bar', orientation='vertical', edgecolor='black', linewidth=1.2)
        if verbosity > 2:
            plt.axvline(np.percentile(simulations.iloc[-1], 5), color='r', linestyle='dashed', linewidth=0.5)
            plt.axvline(np.percentile(simulations.iloc[-1], 95), color='r', linestyle='dashed', linewidth=0.5)
        plt.show()

    if verbosity > 2:
        print(simulations.iloc[-1].describe())
        print('5th percentile\t{}'.format(np.percentile(simulations.iloc[-1], 5)))
        print('95th percentile\t{}'.format(np.percentile(simulations.iloc[-1], 95)))

def lec2_merton_jump_diffusion(risk_free_rate_aka_drift = 0.05,
                                  sigma_aka_volatility = 0.3,
                                  share_price_at_t0 = 20,
                                  lambda_aka_poisson_intensity = 0.95,
                                  verbosity = 2):
    # Let's JUMP!
    # Merton Jump Diffusion Model (MDJ) Model UNDER risk-neutral measure
    # P160 of Ch1_MFIT5003_Fall2020-21_with_MJD.pdf

    T = 2
    #mu = 0.2  # without drift
    # mu is replace by risk_free_rate below
    σ = sigma_aka_volatility # The same as GBM Model
    S0 = share_price_at_t0
    dt = 0.01 # step side
    N = round(T/dt)

    t = np.linspace(0, T, N) # return evenly spaced numbers over a specified inteval

    risk_free_rate = risk_free_rate_aka_drift # risk-free interest rate
    # According to page 156, dNt denotes the number of jumps. dNt cannot be predicted
    # accurately, we assume it follows Poisson distribution. Nt ~ Possion(λt)
    λ = lambda_aka_poisson_intensity

    # jump size. According to page 150 of lec2-3_ch1_introduction_with_mjd
    # Suppose price jumps from St to Jt*St, Js is called an absolute price jump size Merton assumes that Jt
    # is a positive random variable from lognormal distribution logJt ~ N(μ, σ^2).
    # This assumption, however, is more or less a subjective one. Users are free to use any other distributions instead.
    μ_jump = -0.6
    σ_jump = 0.25

    # In MJD Model we have two more random variables compared with Geometric Brownian Motion (GBM) model, namely
    # jump_size (assumed to follow normal distribution) and number_of_jump (assumed to follow Possion distribution)

    k = np.exp(μ_jump + 0.5*σ_jump**2) - 1
    # k is defined as the mean of (Jt - 1), (not Jt -1 itself)

    μ = risk_free_rate - λ * k
    # In this case, interest rate is not equal to mu.

    S = np.zeros(N+1)
    S[0] = S0

    for t in range (1, N+1):
        # Geometric Brownian Motion
        GBM_part = S[t-1] * (np.exp((μ - 0.5 * σ ** 2) * dt + σ * np.random.standard_normal(size=1) * np.sqrt(dt)))
        MDJ_part = S[t-1] * (np.exp(μ_jump + σ_jump * np.random.standard_normal(size=1))-1) * np.random.poisson(λ*dt, size=1)
        S[t] = GBM_part + MDJ_part
    if verbosity > 0:
        S = pd.DataFrame(S)
        S.plot(figsize=(10, 7), grid = True, legend = False)
        plt.title('Stimulation of Merton Jump Diffusion Model')
        plt.xlabel('Time')
        plt.ylabel('Price')

    return S

def lec2_monte_carlo_simulation_with_mjd(risk_free_rate_aka_drift = 0.05,
                                  sigma_aka_volatility = 0.3,
                                  share_price_at_t0 = 20,
                                  lambda_aka_poisson_intensity = 0.95,
                                  number_of_simulations = 100,
                                  number_of_bins = 20,
                                  verbosity = 2):

    B = {}

    for i in range(number_of_simulations):
        B['Simulation {}'.format(i)] = lec2_merton_jump_diffusion(
                                  risk_free_rate_aka_drift = risk_free_rate_aka_drift,
                                  sigma_aka_volatility = sigma_aka_volatility,
                                  share_price_at_t0 = share_price_at_t0,
                                  lambda_aka_poisson_intensity = lambda_aka_poisson_intensity,
                                  verbosity = 0)
    B = pd.DataFrame(B)
    if verbosity > 0:
        B.plot(figsize=(10, 7), grid = True, legend = False)
        plt.title('Plot of {} simulations of Merton Jump Diffusion Model'.format(number_of_simulations))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()

    if verbosity > 1:
        plt.hist(x=B.iloc[-1], bins=number_of_bins, histtype='bar', orientation='vertical', edgecolor='black', linewidth=1.2)
        if verbosity > 2:
            plt.axvline(np.percentile(B.iloc[-1], 5), color='r', linestyle='dashed', linewidth=0.5)
            plt.axvline(np.percentile(B.iloc[-1], 95), color='r', linestyle='dashed', linewidth=0.5)
        plt.show()

    if verbosity > 2:
        print(B.iloc[-1].describe())
        print('5th percentile\t{}'.format(np.percentile(B.iloc[-1], 5)))
        print('95th percentile\t{}'.format(np.percentile(B.iloc[-1], 95)))

def lec3_students_t_test_for_population_mean(dataset: np.ndarray,
                                             proposed_population_mean: float,
                                             significance_level: float,
                                             normal_distribution_assumption_met: bool,
                                             verbosity = 2):
    """ One-sample T Test (also called a two-sided test):
    SciPy Manual: This is a two-sided test for the null hypothesis that the expected value (mean) of
    a sample of independent observations a is equal to the given population mean, popmean.

    refer to pages 190 - 192 of lec2-3_ch1_introduction_with_mjd
    """

    assert np.isnan(dataset).any() == False
    assert significance_level < 0.2
    assert normal_distribution_assumption_met == True # This method requires normal distribution assumption


    α = significance_level
    μ = proposed_population_mean
    confidence_level = 1 - α
    assert α == significance_level
    assert confidence_level == 1 - significance_level

    from scipy.stats import ttest_1samp
    stat, p = ttest_1samp(a=dataset, popmean=μ)
    p_left = 1 - p / 2 if stat > 0 else p / 2
    p_right = p / 2 if stat > 0 else 1 - p / 2

    if verbosity > 0:
        print('Student\'s two-sided t test result:')
        print('p == {}'.format(round(p, 10)), end='')
        if p > α:
            print(' > {}, do not reject H0 (that population mean (μ) == {})'.format(α, μ))
        else:
            print(' <= {}, reject H0 (that population mean (μ) == {})'.format(α, μ))
        print('sample mean: {}\n'.format(np.array(dataset).mean()))

    if verbosity > 1:
        print('Student\'s one-sided t test result:')
        print('p_left == {}'.format(p_left), end = '')
        if p_left > α:
            print(' > {}, do not reject H0 (that population mean (μ) >= {})'.format(α, μ))
        else:
            print(' <= {}, reject H0 (that population mean (μ) >= {})'.format(α, μ))

        print('p_right == {}'.format(p_right), end = '')
        if p_right > α:
            print(' > {}, do not reject H0 (that population mean (μ) <= {})'.format(α, μ))
        else:
            print(' <= {}, reject H0 (that population mean (μ) <= {})\n'.format(α, μ))


def lec3_confidence_interval_for_weight_mean(dataset: np.ndarray,
                                             confidence_level: float,
                                             normal_distribution_assumption_met: bool):
    """
    arguments:
    confidence_level -- In plain English, a Confidence Interval is a range of values we are fairly sure our true value lies in.
     The level of "fair surety" is called confidence level significance level (alpha) + confidence level = 1
     alpha is also the threshold of pvalue.
    """

    assert np.isnan(dataset).any() == False
    assert confidence_level > 0.8
    assert normal_distribution_assumption_met == True

    α = 1 - confidence_level
    ci_lower_bound, ci_upper_bound = sms.DescrStatsW(dataset).tconfint_mean(alpha=α)
    print('C.I. with {}% confidence: [{}, {}]\n'.format(confidence_level * 100, round(ci_lower_bound, 10), round(ci_upper_bound, 10)))
    return ci_lower_bound, ci_upper_bound

def lec3_wilcoxon_signed_rank_median_test(dataset: np.ndarray,
                                          hypothesized_median: float,
                                          significance_level: float,
                                          apply_correction_to_discrete_values: bool,
                                          normal_distribution_assumption_met: bool,
                                          verbosity = 2):
    """
    Basically you try to determine the probability that the median of a dataset is hypothesized_median.
    Continous or discrete? check this link: https://en.wikipedia.org/wiki/Continuous_or_discrete_variable
    """

    assert significance_level < 0.2
    assert isinstance(dataset, np.ndarray)

    α = significance_level
    stat1, p1 = sp.stats.wilcoxon(dataset - hypothesized_median,
                                  correction = apply_correction_to_discrete_values)
    stat2, p2 = sp.stats.wilcoxon(dataset,
                                  np.array([hypothesized_median] * len(dataset)),
                                  correction = apply_correction_to_discrete_values)

    assert stat1 == stat2 and p1 == p2

    if verbosity > 0:
        print('Wilcoxon signed-rank test result:')
        print('p == {}, '.format(p1), end = '')
        if p1 > α:
            print('do NOT reject H0 (that median == {})'.format(hypothesized_median))
        else:
            print('reject H0 (that median == {})'.format(hypothesized_median))

    if verbosity > 1:
        print('NOTES:\n(1) Dr. Yu\'s solution listed both corrected an uncorrected values of wilcoxon test. Therefore, perhaps correction is not a big issue for the purpose of this course.')
        if normal_distribution_assumption_met == True:
            print('(2) This test does NOT require normal distribution assumption!')


def lec3_skewness_test(dataset: np.ndarray,
                       significance_level = 0.05,
                       verbosity = 2):

    assert isinstance(dataset, np.ndarray)
    assert np.isnan(dataset).any() == False # One nan can contaminate the entire list!
    assert len(dataset) > 7 # refer to page 205 of lec2-3_ch1_introduction_with_mjd
    assert significance_level < 0.2

    α = significance_level
    stat, p = sp.stats.skewtest(dataset)
    p_left = 1 - p / 2 if stat > 0 else p / 2
    p_right = p / 2 if stat > 0 else 1 - p / 2

    if verbosity > 0:
        print('Skewness two-sided test result:')
        print('p == {}'.format(p), end='')
        if p > α:
            print(' > {}, do not reject H0 (that the skewness of the population that the sample was drawn from is the same as that of a corresponding normal distribution)'.format(α))
        else:
            print(' <= {}, reject H0 (that the skewness of the population that the sample was drawn from is the same as that of a corresponding normal distribution)'.format(α))

    if verbosity > 1:
        print('Skewness one-sided test result:')
        print('p_left == {}'.format(p_left), end = '')
        if p_left> α:
            print(' > {}, do not reject H0 (that the distribution generating our data has an insignificantly long tail on the LEFT)'.format(α))
        else:
            print(' <= {}, reject H0 (that the distribution generating our data has an insignificantly long tail on the LEFT)'.format(α))

        print('p_right == {}'.format(p_right), end = '')
        if p_right > α:
            print(' > {}, do not reject H0 (that the distribution generating our data has an insignificantly long tail on the RIGHT)'.format(α))
        else:
            print(' <= {}, reject H0 (that the distribution generating our data has an insignificantly long tail on the RIGHT)\n'.format(α))

def lec3_kurtosis_test(dataset: np.ndarray,
                       significance_level = 0.05,
                       verbosity = 2):

    assert isinstance(dataset, np.ndarray)
    assert significance_level < 0.2
    assert np.isnan(dataset).any() == False # One nan can contaminate the entire list!

    α = significance_level
    stat, p = sp.stats.kurtosistest(dataset)
    p_left = 1 - p / 2 if stat > 0 else p / 2
    p_right = p / 2 if stat > 0 else 1 - p / 2
    if verbosity > 0:
        print('Kurtosis two-sided test result:')
        print('p == {}'.format(p), end='')
        if p > α:
            print(' > {}, do not reject H0 (that the distribution generating our data has INsignificantly different kurtosis from a normal distribution)'.format(α))
        else:
            print(' <= {}, reject H0 (that the distribution generating our data has INsignificantly different kurtosis from a normal distribution)'.format(α))

    if verbosity > 1:
        print('Kurtosis one-sided test result:')
        print('p_left == {}'.format(p_left), end = '')
        if p_left > α:
            print(' > {}, do not reject H0 (that the distribution generating our data has INsignificantly thinner tails than a normal distribution)'.format(α))
        else:
            print(' <= {}, reject H0 (that the distribution generating our data has INsignificantly thinner tails than a normal distribution)'.format(α))

        print('p_right == {}'.format(p_right), end = '')
        if p_right > α:
            print(' > {}, do not reject H0 (that the distribution generating our data has INsignificantly thicker tails than a normal distribution)'.format(α))
        else:
            print(' <= {}, reject H0 (that the distribution generating our data has INsignificantly thicker tails than a normal distribution)\n'.format(α))

def lec3_moods_2sample_same_variance_test(dataset1: np.ndarray,
                                          dataset2: np.ndarray,
                                          significance_level: float,
                                          both_datasets_from_normal_distribution: bool):
    """
    Mood’s two-sample test for scale parameters is a test for the null hypothesis that two
    samples are drawn from the same distribution with the same scale parameter.
    Wikipedia: It tests the null hypothesis that the medians of the populations from which two or more samples are drawn are identical.

    H0: σ1^2 / σ2^2 == 1

    """
    assert isinstance(dataset1, np.ndarray)
    assert isinstance(dataset2, np.ndarray)
    assert significance_level < 0.2
    assert both_datasets_from_normal_distribution

    α = significance_level
    z, p = sp.stats.mood(dataset1, dataset2)

    print('Mood’s two-sample equal variance test result:\np == {}, '.format(p), end='')
    if p > α:
        print('variances of two dataset look INsignificantly different (fail to reject H0)\n')
    else:
        print('variances of two dataset look significantly different (reject H0)\n')
   # print('A more detailed null hypothesis: two samples are drawn from the same distribution with the same scale parameter')

def lec3_ttest_2sample_with_same_mean(dataset1: np.ndarray,
                                      dataset2: np.ndarray,
                                      significance_level: float,
                                      normal_distribution_assumption_met: bool,
                                      independent_assumption_met: bool,
                                      equal_variance_assumption_met: bool):
    """
    null hypothesis that 2 independent samples have identical average (expected) values.
    """

    assert isinstance(dataset1, np.ndarray)
    assert isinstance(dataset2, np.ndarray)
    assert significance_level < 0.2
    assert normal_distribution_assumption_met # If the assumption is not met, use lec3_wilcoxon_rank_sum_2sample_median_test instead!
    assert independent_assumption_met
    # In regards to what exactly does independent mean, refer to page 219 of lec2-3_ch1_introduction_with_mjd
    #assert equal_variance_assumption_met

    print('It appears that this test does NOT need the assumption of normal distribution! The original implementation was WRONG!')
    
    α = significance_level

    # How to determine equal_variance_assumption? Use lec3_moods_2sample_same_variance_test!
    stat, p = sp.stats.ttest_ind(dataset1, dataset2, equal_var = equal_variance_assumption_met)
    print('T-test_ind result:\np == {}'.format(p), end='')
    if p > α:
        print(' > {}, fail to reject H0 (that the population means of two datasets are equal)'.format(α))
    else:
        print(' <= {}, reject H0 (that the population means of two datasets are equal)'.format(α))

    cm = sms.CompareMeans(sms.DescrStatsW(dataset1), sms.DescrStatsW(dataset2))
    # note sms.DescrStatsW().tconfint_mean() and sms.DescrStatsW() are DIFFERENT!
    ci_lower, ci_upper = cm.tconfint_diff(alpha = α, usevar = 'pooled' if equal_variance_assumption_met else 'unequal')
    print('sms.CompareMeans result:\n[{}, {}]'.format(ci_lower, ci_upper), end='')
    if 0  > ci_lower and 0 < ci_upper:
        print(': 0 is in the confidence interval at {}% confidence level\n'.format((1 - α) * 100))

def lec3_wilcoxon_rank_sum_2sample_median_test(dataset1: np.ndarray,
                                               dataset2: np.ndarray,
                                               significance_level: float,
                                               normal_distribution_assumption_met = False):
    """
    My understanding is that the key difference between wilcoxon_rank_sum_2samples_test and moods_2samples_median_test is that
    # mood's test assumes normal distribution (pages 218 and 220)
    while wilcoxon's rank sum test does not need this assumption
    (page 222 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf).
    """

    assert isinstance(dataset1, np.ndarray)
    assert isinstance(dataset2, np.ndarray)
    assert significance_level < 0.2
    assert normal_distribution_assumption_met == False # If the assumption is met, should use lec3_ttest_2sample_with_same_mean instead!

    α = significance_level

    stat, p = sp.stats.ranksums(dataset1, dataset2)
    print('Wilcoxon rank-sum test retuls\np == {}'.format(p), end='')
    if p > α:
        print(' > {}, medians from two samples look INsignificantly different (fail to reject H0)'.format(α))
    else:
        print(' <= {}, medians from two samples look significantly different (reject H0)'.format(α))
    print('Note: It tests whether two samples are likely to derive from the same population. Some investigators interpret this test as comparing the medians between the two populations.\n')

def lec3_shapiro_wilk_normality_test(dataset: np.ndarray,
                                     significance_level: float):
    """
    Refer to pages 228-229 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
    """

    assert isinstance(dataset, np.ndarray)
    assert significance_level < 0.2
    assert len(dataset) >= 3 and len(dataset) <= 5000   # Refer to page 228 of lec2-3_ch1_introduction_with_mjd
    assert np.isnan(dataset).any() == False

    α = significance_level
    confidence_level = 1 - α
    assert confidence_level == 1 - α

    stat, p = sp.stats.shapiro(dataset)
    print('Shapiro-Wilk test result:\np == {}'.format(p), end='')
    if p > α:
        print(' (p > {}), sample looks Gaussian (fail to reject H0)\n'.format(α))
    else:
        print(' (P <= {}), sample does NOT look Gaussian (reject H0)\n'.format(α))


def lec3_kolmogorov_smirnov_normality_test(dataset: np.ndarray,
                                        dataset_standardized: bool,
                                        significance_level: float):
    """
    Refer to pages 228-229 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf
    """

    assert significance_level < 0.2
    assert isinstance(dataset, np.ndarray)
    assert np.isnan(dataset).any() == False

    α = significance_level

    # According to page 229 of lec2-3_ch1_introduction_with_mjd.pdf, dataset for KS test has to be standardized.
    if dataset_standardized == False:
        dataset = (dataset - np.mean(dataset))/np.std(dataset)
   #     print('IMPORTANT: dataset will be STANDARDIZED if it has not been done!')

    stat, p = sp.stats.kstest(dataset, cdf = 'norm')
    print('Kolmogorov-Smirnov test result:\np == {},'.format(p), end='')
    if p > α:
        print(' (p > {}), sample looks Gaussian (fail to reject H0)\n'.format(α))
    else:
        print(' (p <= {}), sample does NOT look Gaussian (reject H0)\n'.format(α))

def lec3_anderson_darling_normality_test(dataset: np.ndarray):
    """
    Refer to pages 228-229 of lecture-2-3_Ch1_MFIT5003_Fall2020-21_with_MJD.pdf

    anderson-darling test does not provide a concrete p value
    """

    assert isinstance(dataset, np.ndarray)
    assert np.isnan(dataset).any() == False
    assert len(dataset) > 7 # Refer to page 228 of lec2-3_ch1_introduction_with_mjd

    stat, critical_values, significance_levels = sp.stats.anderson(x = dataset, dist = 'norm')
    print('Anderson-Darling test results:\nstats: {}\ncritical_values: {}\nsignificance_levels: {}'.format(stat, critical_values, significance_levels))

    print('Results interpretation:')
    p_upper, p_lower = 2147483648, -2147483648
    for i in range(len(critical_values)):
        if stat > critical_values[i]:
            if p_upper > (significance_levels[i] / 100):
                p_upper = significance_levels[i] / 100
            print('sample does NOT look Gaussian (reject H0) (p < alpha == {})'.format(significance_levels[i] / 100))
        else:
            if p_lower < (significance_levels[i] / 100):
                p_lower = significance_levels[i] / 100

            print('sample looks Gaussian (fail to reject H0) (p > alpha == {})'.format(significance_levels[i] / 100))
    print('p ∈ ({}, {})\n'.format(p_lower, p_upper)) # Not sure whether an open or a closed interval should be used.

def lec4_scatter_plot_and_pearson_correlation_coefficient(
            dataset_x: Union[np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series], 
            dsx_name: str,
            dataset_y: Union[np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series], 
            dsy_name: str,
            bivariate_normal_distribution_assumption_met = True,
            model_for_predictions = None,
            dsx_name_in_patsy_formula = None):
    
    assert np.isnan(dataset_x).any() == False and np.isnan(dataset_y).any() == False
    assert isinstance(dataset_x, (list, np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series))
    assert isinstance(dataset_y, (list, np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series))
    
    fig = plt.figure(figsize = (4 * 3, 3 * 3))

    fig.add_subplot(2, 2, 1)
    fig.canvas.set_window_title('scatter_plot_and_subplot')

    # "2,2,1" means "2x2 grid, 1st subplot".
    plt.hist(dataset_x, bins = 20, color = 'green', alpha = 0.9)
    plt.title(dsx_name)

    fig.add_subplot(2, 2, 3)
    plt.scatter(dataset_x, dataset_y, marker = '+', alpha = 0.9)
    if model_for_predictions is not None and dsx_name_in_patsy_formula is not None:
        styles = ['r--', 'g--', 'r--', 'g--']
        assert len(model_for_predictions) < 5
        for i in range(len(model_for_predictions)): 
            predictions = model_for_predictions[i].predict(exog={dsx_name_in_patsy_formula[i]: dataset_x})
            plt.plot(dataset_x, predictions, styles[i], linewidth = 2, label = f'Predicted ({i})')
        
    fig.add_subplot(2, 2, 4)
    plt.hist(dataset_y, bins = 20, orientation = 'horizontal', color = 'red', alpha = 0.9)
    plt.title(dsy_name)
    
    plt.show() 

    retval = np.corrcoef(dataset_x, dataset_y)
    r2, _ = sp.stats.stats.pearsonr(dataset_x, dataset_y)
    assert abs(retval[0][0] - retval[1][1]) < 10 ** -10
    assert abs(retval[1][0] - retval[0][1]) < 10 ** -10
    r1 = retval[1][0]

    assert r1 - r2 < 10 ** -10

    print('Pearson\'s correlation coefficient:\n{}, '.format(retval[1][0]), end = '')
    if r1 > 0.7:
        print('indicating a strong POSITIVE linear correlation between [{}] and [{}]'.format(dsx_name, dsy_name))
    elif r1 < -0.7:
        print('indicating a strong NEGATIVE linear correlation between [{}] and [{}] '.format(dsx_name, dsy_name))
    else:
        print('indicating NO significant correlation between [{}] and [{}]'.format(dsx_name, dsy_name))
    print('r^2 == {}%, meaning that aroung {}% of the variation of [{}] can be explained by the variation of [{}]'.format(round(r1 ** 2 * 100, 1), round(r1 ** 2 * 100, 1), dsy_name, dsx_name))
    if bivariate_normal_distribution_assumption_met == False:
        print('NOTE: If bivariate normal distribution assumption is NOT met, you should consider using Spearman\'s rank correlation cofficient or Kendall\'s rank correlation coefficient instead!')
    print('')
    
def lec4_spearmans_rank_correlation_and_kendalls_rank_correlation(
        dataset1: np.ndarray, 
        ds1_name: str,
        dataset2: np.ndarray, 
        ds2_name: str,
        bivariate_normal_distribution_assumption_met = False):
    '''
    Refer to page 15 of lec4-5_ch2-linear-regression-models
    '''

    assert isinstance(dataset1, np.ndarray) and isinstance(dataset1, np.ndarray)
    assert np.isnan(dataset1).any() == False
    assert np.isnan(dataset2).any() == False

    significance_level = 0.05

    correlation_coefficient, p = sp.stats.spearmanr(dataset1, dataset2)
    print('Results from Spearman\'s rank test (in particular for ordinal variables): p == {}, '.format(p), end='')
    if p > significance_level:
        print(' (p > {}), fail to reject H0 that there is no monotonic relatinship between two variables'.format(significance_level))
    else:
        print(' (p <= {}), reject H0 that there is no monotonic relatinship between two variables'.format(significance_level))
    print('correlation_coefficient == {}, '.format(correlation_coefficient), end = '')
    if correlation_coefficient > 0.5:
        print('indicating a strong POSITIVE linear/non-linear correlation between [{}] and [{}]'.format(ds1_name, ds2_name))
    elif correlation_coefficient < -0.5:
        print('indicating a strong NEGATIVE linear/non-linear correlation between [{}] and [{}] '.format(ds1_name, ds2_name))
    else:
        print('indicating NO significant correlation between [{}] and [{}]'.format(ds1_name, ds2_name))

    correlation_coefficient, p = sp.stats.kendalltau(dataset1, dataset2)
    print('\nResults from Kendall\'s rank test (in particular for ordinal variables): p == {}'.format(p), end = '')
    if p > significance_level:
        print(' (p > {}), fail to reject H0 that there is no monotonic relatinship between two variables'.format(significance_level))
    else:
        print(' (p <= {}), reject H0 that there is no monotonic relatinship between two variables'.format(significance_level))
    print('correlation_coefficient == {}, '.format(correlation_coefficient), end = '')
    if correlation_coefficient > 0.5:
        print('indicating a strong POSITIVE linear/non-linear correlation between [{}] and [{}]'.format(ds1_name, ds2_name))
    elif correlation_coefficient < -0.5:
        print('indicating a strong NEGATIVE linear/non-linear correlation between [{}] and [{}] '.format(ds1_name, ds2_name))
    else:
        print('indicating NO significant correlation between [{}] and [{}]'.format(ds1_name, ds2_name))

    if bivariate_normal_distribution_assumption_met == True:
        print('NOTE: Bivariate normal distribution assumption does NOT prevent users from using this method. They should, however, consider using lec4_scatter_plot_and_pearson_correlation_coefficient() as well!')

def lec4_ols_model_helper(
    patsy_formula: str,
    variable_names_and_values: Dict[str, np.ndarray],
    with_intercept: bool,
    verbosity: int,
    significance_level = 0.05, 
    independence_constant_variance_zero_mean_and_normal_distritbuion_assumptions_met = True):

    assert significance_level < 0.2
    assert independence_constant_variance_zero_mean_and_normal_distritbuion_assumptions_met
    # Refer to page 54 of lec4-5_ch2-linear-regression-models
    patsy_formula = patsy_formula.replace(' ', '')
    assert (('-1' not in patsy_formula) == with_intercept)
    # This assertion is not 100% reliable. But for the purpose of this course it should be enough
    α = significance_level

    model = smf.ols(formula = patsy_formula, data = variable_names_and_values).fit()
    # https://www.statsmodels.org/stable/generated/statsmodels.formula.api.ols.html
    
    assert model.params.keys()[0] != 'Intercept' or with_intercept
    # This assertion is not 100% reliable since users can name a variable "Intercept"
    
    if verbosity > 0:
        print(model.summary(alpha = α))
        print('\n')

    if verbosity > 1:
        print('                    === OLS Regression Results Interpretation ===')
        print('Goodness indicators:')
        print('R²: {} (closely related to Pearson correlation²)\nAdjusted R²: {} (Used to compare nested models)'.format(round(model.rsquared, 8), round(model.rsquared_adj, 8)))
        print('AIC: {}'.format(model.aic))
        print('BIC: {}'.format(model.bic))
        print('σ_hat of residuals: {}'.format(model.mse_resid ** 0.5))
        
        ci_results = model.conf_int(alpha=α, cols=None)
        
        print('\nRegression coefficients:')

        for i in range(len(model.params)):
            print('β{} == {} ∈ [{}, {}] at {}%'.format( 
                str(i if with_intercept else i + 1).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")),
                round(model.params[i], 8),
                round(ci_results[0][i], 8), 
                round(ci_results[1][i], 8),
                (1 - α) * 100))


        for i in range(len(model.pvalues)):
            print('p-value (H₀: β{} == 0) == {} {}'.format(
                str(i if with_intercept else i + 1).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")), 
                round(model.pvalues[i], 8),
              f'> {α}, do NOT reject H₀' if model.pvalues[i] > α else f'<= {α}, reject H₀'))
        
        if with_intercept:
            if model.pvalues[0] > α:
                print('\nIMPORTANT NOTE: since β₀ is INsignificantly different from 0, we should DROP the intercept!')
            else:
                print('\nNOTE: since β₀ is significantly different from 0, we canNOT drop the intercept!')

    if verbosity > 0:
        print('\n\n')
    return model

def lec4_test_β_at_specific_values(model, 
                                   matrix_a, 
                                   vector_b, 
                                   significance_level = 0.03):
    """
    Formula of H₀: ax = b
    Example I (with intercept):
        to test β₀ == 0, β₁ == 0, β₂ == 0
        [[1, 0, 0]
         [0, 1, 0]  * [β₀, β₁, β₂]ᵀ = [0, 0, 0]ᵀ
         [0, 0, 1]]
    Example II (with intercept):
        to test β₁ == 1
        [0, 1] * [β₀, β₁]ᵀ = 1
    Example III (with intercept):
        to test β₁ == β₂
        [0, 1, -1]* [β₀, β₁, β₂]ᵀ = 0
    Example IV (withOUT intercept):
        to test β₁ == β₂
        [1, -1]* [β₁, β₂]ᵀ = 0
    Example V (withOUT intercept):
        to test β₁ == 1
        1 * β₁ = 1

    """
    print('β values hypothesis test:')
    print('Matrix A: {}'.format(matrix_a))
    print('Vector b: {}'.format(vector_b))
    if (isinstance(vector_b, list) and len(vector_b)) or isinstance(vector_b, (int, float)):
        print('NOTE: It appears that if vector_b has only one component, passing a list or a number are both fine.')
    
    B = (matrix_a, vector_b)
    p = model.f_test(B).pvalue

    print('Results from F-Test: p == {} '.format(p), end='')
    if p > significance_level:
        print('(p > {}), fail to reject H₀ '.format(significance_level))
    else:
        print('(p <= {}), reject H₀'.format(significance_level))
    print('NOTE: Due to the complexity of this test, no natural language interpretation is generated. Use these keywords to draw a conclusion: significant/insignificantly different, reject/fail to reject H₀.\n')
    
def lec4_predict_y_with_model(model,
                         intercept_included: bool,
                         independent_variables: List[np.ndarray],
                         iv_names: List[str],
                         significance_level = 0.1):
    """
    Format of independent_variables:
      x1  [[1, 2, 3, 4, 5],
      x2   [5, 4, 3, 2, 1],
      x3   [6, 6, 6, 6, 6]]
    """
    ivs = independent_variables
    assert significance_level < 0.2
    assert model.params.keys()[0] != 'Intercept' or intercept_included
    # This assertion is not 100% reliable since users can name a variable "Intercept"
    α = significance_level

    print('Predicting y with model (α == {}, {}):'.format(α, 'intercept included' if intercept_included else 'intercept EXCLUDED'))

    model_parameters = {}
    if intercept_included:
        model_parameters['intercept'] = 1
    for i in range(len(ivs)):
        model_parameters[iv_names[i]] = ivs[i]
    print(model_parameters)
    predicted_values = model.predict(pd.DataFrame(model_parameters))

    results = []
    for i in range(len(ivs[0])):
        tmp = []
        for j in range(len(iv_names)):
            tmp.append(ivs[j][i])
        tmp.append(predicted_values[i])
        results.extend([tmp])

    results_pd = pd.DataFrame(results)
    headers = []
    for i in range(len(iv_names)):
        headers.append(iv_names[i])
    headers.append('Predicted Value')

    results_pd.columns = headers

    exogenous_parameters = []
    for i in range(len(ivs[0])):
        temp_iv = [1] if intercept_included else []
        # The first [1] means intercept is included.
        for j in range(len(ivs)):
            temp_iv.append(ivs[j][i])
        temp_iv = np.array(temp_iv)
        exogenous_parameters.append(temp_iv)
    exogenous_parameters = np.array(exogenous_parameters)
    if intercept_included == False:
        exogenous_parameters = exogenous_parameters.reshape(-1, 1)
        # https://stackoverflow.com/questions/51507423/statsmodels-return-prediction-interval-for-linear-regression-without-an-interce/51519526

    results_pd['Prediction Std'], results_pd['L-Bound'], results_pd['U-Bound'] = wls_prediction_std(model, exog = exogenous_parameters, weights = 1, alpha = α)

    print(results_pd)
    if intercept_included == False:
        print('NOTE: confidence level prediction for model withOUT intercept should work but its correctness is not thoroughly tested!')
    print('')
    
def __lec5_model_evaluator(dataset: np.ndarray,
                    column_names_of_explanatory_aka_independent_variables: List[str],
                    column_name_of_response_aka_dependent_variable: str):

    x_names = column_names_of_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable
    y = dataset[y_name]

    model = sm.OLS(y, sm.add_constant(dataset[list(x_names)]))
    # add_constant: intercept included
    regr = model.fit()

    return {'model': regr,
            'variables': list(x_names),
            'aic': regr.aic,
            'bic': regr.bic,
            'r2': regr.rsquared,
            'adj_r2': regr.rsquared_adj}

def __lec5_forward_selector(
                     dataset: np.ndarray,
                     column_names_of_USED_explanatory_aka_independent_variables: List[str],
                     column_names_of_ALL_explanatory_aka_independent_variables: List[str],
                     evaluation_criterion: str,
                     column_name_of_response_aka_dependent_variable: str):

  #  print(f'USED: {column_names_of_USED_explanatory_aka_independent_variables}')
  #  print(f'ALL: {column_names_of_ALL_explanatory_aka_independent_variables}')
    criterion = evaluation_criterion.lower()
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')

    used_x_names = column_names_of_USED_explanatory_aka_independent_variables
    x_names = column_names_of_ALL_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable

    unused_x_names = [new_x for new_x in x_names if new_x not in used_x_names]
    results = []

    for new_x in unused_x_names:
        results.append(__lec5_model_evaluator(dataset = dataset,
                    column_names_of_explanatory_aka_independent_variables = used_x_names + [new_x],
                    column_name_of_response_aka_dependent_variable = y_name))
   #     print(results[len(results) - 1]['variables'])

    models = pd.DataFrame(results)

    if criterion == 'aic' or criterion == 'bic':
        model_each = models.loc[models[criterion].idxmin()]
    else:
        model_each = models.loc[models[criterion].idxmax()]
    return model_each

def lec5_get_best_model_with_forward_selection(
        evaluation_criterion: str,
        dataset: pd.DataFrame,
        column_names_of_explanatory_aka_independent_variables: List[str],
        column_name_of_response_aka_dependent_variable: str):

    criterion = evaluation_criterion.lower()
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')
    assert isinstance(dataset, pd.core.frame.DataFrame)
    print(f'=== Picking model by [{criterion}] with FORWARD sequential selection ===')

    y_name = column_name_of_response_aka_dependent_variable
    x_names = column_names_of_explanatory_aka_independent_variables

    best_models = pd.DataFrame(columns = ['model', 'aic', 'bic','r2', 'adj_r2', 'variables'])
  #  best_models = pd.DataFrame(columns = [criterion, 'variables'])
    exp = []

    for i in range(1, len(x_names) + 1):
        best_models.loc[i] = __lec5_forward_selector(dataset = dataset,
                  column_names_of_USED_explanatory_aka_independent_variables = exp,
                  column_names_of_ALL_explanatory_aka_independent_variables = x_names,
                  evaluation_criterion = criterion,
                  column_name_of_response_aka_dependent_variable = y_name)
        exp = best_models.loc[i]['variables']
    if criterion == 'aic' or criterion == 'bic':
        best_models = best_models.sort_values(by = criterion, ascending = True)
    else:
        best_models = best_models.sort_values(by = criterion, ascending = False)

    print(best_models[[criterion, 'variables']])
    print('')
    return best_models

def lec5_get_best_model_with_backward_elimination(
        dataset: pd.DataFrame,
        evaluation_criterion: str,
        column_names_of_explanatory_aka_independent_variables: List[str],
        column_name_of_response_aka_dependent_variable: str):

    criterion = evaluation_criterion.lower()
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')
    assert isinstance(dataset, pd.core.frame.DataFrame)

    print(f'=== Picking model by [{criterion}] with BACKWARD sequential elimination ===')
    x_names = column_names_of_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable

    best_models = pd.DataFrame(columns = ['model', 'aic', 'bic','r2', 'adj_r2', 'variables'])

    best_models.loc[len(x_names)] = __lec5_model_evaluator(dataset = dataset,
                    column_names_of_explanatory_aka_independent_variables = x_names,
                    column_name_of_response_aka_dependent_variable = y_name)

    while len(x_names) > 1:
        best_models.loc[len(x_names) - 1] = __lec5_backward_eliminator(
                    dataset = dataset, evaluation_criterion = criterion,
                    column_names_of_explanatory_aka_independent_variables = x_names,
                    column_name_of_response_aka_dependent_variable = y_name)
        x_names = best_models.loc[len(x_names) - 1]['variables']

    if criterion == 'aic' or criterion == 'bic':
        best_models = best_models.sort_values(by = criterion, ascending = True)
    else:
        best_models = best_models.sort_values(by = criterion, ascending = False)
    print(best_models[[criterion, 'variables']])
    print('')
    return best_models

def __lec5_backward_eliminator(dataset: np.ndarray, evaluation_criterion: str,
                    column_names_of_explanatory_aka_independent_variables: List[str],
                    column_name_of_response_aka_dependent_variable: str):

    criterion = evaluation_criterion
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')
    x_names = column_names_of_explanatory_aka_independent_variables
    results = []
    for combo in itertools.combinations(x_names, len(x_names) - 1):
        results.append(__lec5_model_evaluator(dataset = dataset,
                    column_names_of_explanatory_aka_independent_variables = combo,
                    column_name_of_response_aka_dependent_variable = column_name_of_response_aka_dependent_variable))
        models = pd.DataFrame(results)
        if criterion == 'bic' or criterion == 'aic':
            model_each = models.loc[models[criterion].idxmin()]
        else:
            model_each = models.loc[models[criterion].idxmax()]
    return model_each


    

def lec5_get_best_model_with_enumeration(evaluation_criterion: str,
                      dataset: pd.DataFrame,
                      column_names_of_explanatory_aka_independent_variables: List[str],
                      column_name_of_response_aka_dependent_variable: str):
    # This is called "Best Subset Selection approach" by Dr. Yu.

    criterion = evaluation_criterion.lower()
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')    
    assert isinstance(dataset, pd.core.frame.DataFrame)

    print(f'=== Picking model by [{criterion}] with ENUMERATION ===')
    x_names = column_names_of_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable


    best_models = [None] * (len(x_names) + 1)

    for i in range(1, len(x_names) + 1):
        results = []
        for x_names_combination in itertools.combinations(x_names, i):
            results.append(__lec5_model_evaluator(dataset = dataset,
                    column_names_of_explanatory_aka_independent_variables = x_names_combination,
                    column_name_of_response_aka_dependent_variable = y_name))

        if criterion == 'bic' or criterion == 'aic':
            # AIC and BIC can be considered as measures of error.
            # Therefore, the smaller the better.
            temp_min = 2333333
            for j in range(len(results)):
                if results[j][criterion] < temp_min:
                    best_models[i] = results[j]
                    temp_min = results[j][criterion]
        else:
            temp_max = -2333333
            for j in range(len(results)):
                if results[j][criterion] > temp_max:
                    best_models[i] = results[j]
                    temp_max = results[j][criterion]
    best_models.pop(0)

    best_models = pd.DataFrame(best_models)

    if criterion == 'aic' or criterion == 'bic':
        best_models = best_models.sort_values(by = criterion, ascending = True)
    else:
        best_models = best_models.sort_values(by = criterion, ascending = False)
    best_models = best_models.reset_index(drop=True)
    print(best_models[[criterion, 'variables']])
    if criterion == 'r2':
        print('NOTE: Although nothing prevents you from using r2 as the criterion, r2 almost always ranks the model with the largest number of independent variable the highest.')
    print('')
    return best_models

def __lec5_externally_studentized_residual_plotter(model):
    
    extresid = model.get_influence().resid_studentized_external
    pred = model.predict()
    plt.scatter(pred, extresid, s = 1)
    plt.xlabel('ŷ (the predicted dependent variable)')
    plt.ylabel('ri (the externally studentized residual(i.e. y[i] - ŷ[i]))')
    plt.axhline(y=2, color='g', linestyle='-.', linewidth = 0.5)
    plt.axhline(y=0, color='r', linestyle='-.', linewidth = 0.5)
    plt.axhline(y=-2, color='g', linestyle='-.', linewidth = 0.5)
    plt.show()
    print('Interpretation of the plot:\nIf:')
    print('(1) The plot has no pattern (for possible patterns and their implications, refer to pages 28 - 30 of lec5-6_ch2-variable-selection-and-model-diagnostics);')
    print('(2) Points are around 0; and')
    print('(3) Most of the points are inside the band |r[i]| <= 2')
    print('We say that assumptions A1 - A4 are likely to be valid.')
    print('Note:')
    print('(1) If assumptions 3 or 4 is net met, consider using Box-Cox transformation to remedy it.')
    
def __lec5_residual_normality_test(model):
    
    extresid = model.get_influence().resid_studentized_external
    sm.qqplot(data=extresid, dist=sp.stats.distributions.norm, line='45')
    plt.show()
    print('Interpretation: If many points do not fall near the red line then it is likely that assumption 4 is invalid.')
    lec3_shapiro_wilk_normality_test(dataset = extresid,
                                     significance_level = 0.05)    
    lec3_kolmogorov_smirnov_normality_test(dataset = extresid,
                                        dataset_standardized = False,
                                        significance_level = 0.05)
    lec3_anderson_darling_normality_test(dataset = extresid)
    print('Note: If normality assumption is not valie, consider using Box-Cox transformation to correct it\n')

def __lec6_multicollinearity_detector(dataset_with_all_independent_variables: np.ndarray,
                                      intercept_included = True,
                                      verbosity = 2):
    
    if len(dataset_with_all_independent_variables.shape) < 1:
        print('Ad-hoc check: Number of variable not enough to run multicollinearity check')
        return
    
    if intercept_included:
        dataset_with_all_independent_variables['constant'] = 1
        # This constant is added according to page 80 of lec5-6_ch2-variable-selection-and-model-diagnostics
    else:
        print('NOTE: Dr. Yu has never demonstrated the scenario where an intercept is not included. My feeling is that constant is not needed if the model does not include an incerpt. But this is neither proven nor tested.')
    
    if verbosity >= 1:
        print('Detecting multicollinearity using variance inflation factor:')
    for i in range(dataset_with_all_independent_variables.shape[1]):
        vif = variance_inflation_factor(dataset_with_all_independent_variables.values, i)
        if verbosity >= 1:
            print('[{}] vif == {}: '.format(dataset_with_all_independent_variables.columns[i], vif), end = '')
        if verbosity >= 2:
            if vif > 10:
                print('multicollinearity is SERIOUS')
            else:
                print('multicollinearity is NOT that serious')
        elif verbosity >= 1:
            print('')

    if verbosity >= 3:
        print('Notes:')
        print('(1) The term multicollinearity refers to the effect, on the precision of the LS estimators of the regression coefficients, of two or more of the explanatory variables being highly correlated;')
        print('(2) Pearson correlation matrix among all explanatory variables ONLY shows the association between any two variables, ignoring other explanatory variables;')
        print('(3) We can use variance inflation factor (VIF) to measure the level of multicollinearity. A large VIF indicates a sign of serious multicollinearity. No rule of thumbs on numerical values is foolproof , but it is generally believed that if any VIF exceeds 10 , there may be a serious problem with multicollinearity .')
        print('(4) The simplest method to remedy multicollinearity is just dropping the variables with a large VIF one by one. However, if you prefer not to do so, you may also try Ridge regression. Ridge regression is an alternative to least square regression and it is encapsulated as lec6_ridge_regression()')

def __lec6_ccpr_plot(model):
    fig1 = plt.figure(figsize=(20, 10))
    sm.graphics.plot_ccpr_grid(model, fig=fig1)
    plt.show()
    print('Notes:')
    print('1. Component and component-plus-residual (CCPR) plots are used to check if the true relationship between the mean of dependent variable and independent variables is linear (i.e. if they are linear then it is good);');
    print('2. Given the nature of this checker, no natural language interpretation is generated;')
    print('3. The slope of each plot is more or less similar to the beta value in model\'s summary()')
    print('4. Dr. Yu introduced the mathematial foundation of a remedy called Box-Tidwell transformation. But no code is provided for this method.')
    

def __lec6_autocorrelation_test(model):

    dw = np.sum(np.diff(model.resid.values) ** 2.0) / model.ssr
    print('Durbin-Watson statistic: {}'.format(dw), end = '')
    if dw == 4:
        print(', implying a PERFECT NEGATIVE autocorrelation!')
    elif dw < 4 and dw > 2:
        print(', implying a negative autocorrelation')
    elif dw == 2:
        print(', implying NO autocorrelation AT ALL!')
    elif dw > 0 and dw < 2:
        print(', implying a positive autocorrelation')
    elif dw == 0:
        print(', implying a PERFECT POSITIVE autocorrelation')
    else:
        print(', WTF??!! This is IMPOSSIBLE!!!')
    print('Notes:')
    print('(1) Durbin-Watson statistic should be between 0 to 4. 0 means a perfect positive autocorrelation, 2 means no autocorrelation and 4 means a perfect negative autocorrelation.')
    print('(2) Independence requires both no autocorrelation and normality. Durbin-Watson statistic is only about autocorrelation.')
    print('(3) If autocorrelation is detected, consdering calling lec6_autocorrelation_corrector() to correct it.')
    print('(4) Since (no autocorrelation) + normality  -> independence, to meet the assumption of independence, we may need to apply two remedies.')

    # Residual
    et = model.resid
    # 1-lagged et
    et_1 = et.shift(1)

    plt.scatter(et_1[1:], et[1:], s = 1)
    plt.title('Plotting e_t against e_t-1')
    plt.show()
    # Refer to page 59 of lec5-6_ch2-variable-selection-and-model-diagnostics
    return dw

def lec6_six_assumptions_checker(model,
                                 dataset_with_all_independent_variables: np.ndarray,
                                 intercept_included = True):
    
    print('          ===== Six Assumptions Checker =====')
    print('A1: Ɛ₁...Ɛₙ have zero mean')
    print('A2: Ɛ₁...Ɛₙ are independent')
    print('A3: Ɛ₁...Ɛₙ have a common unknown variance σ²')
    print('A4: Ɛ₁...Ɛₙ are normally distributed')
    print('A5: There is no multicollinearity among independent a.k.a. explanatory variable')
    print('A6: The true relationship between the mean of the dependent variable and independent variables is linear\n')

    print('Externally Studentized Residual Plots (A1-4):')
    __lec5_externally_studentized_residual_plotter(model)

    print('\nAutocorrelation Test (A2)')
    __lec6_autocorrelation_test(model)
    
    print('\nResidual Normality Test (A4):')
    __lec5_residual_normality_test(model)
    
    print('\nMulticollinearity Test (A5):')
    __lec6_multicollinearity_detector(
        dataset_with_all_independent_variables = dataset_with_all_independent_variables,
        intercept_included = intercept_included,
        verbosity = 4)
    
    print('\nCCPR Plot (A6):')
    __lec6_ccpr_plot(model = model)
    print('')
    

    
def lec6_box_cox_transformer(lambda1: float, x1: np.ndarray, y1: np.ndarray):

    if lambda1 != 0:
        BCy = (y1 ** lambda1 - 1) / lambda1
    else:
        BCy = np.log(y1)

    BCfmodel = lec4_ols_model_helper(
        patsy_formula = 'BCy ~ x1',
        variable_names_and_values = {'BCy': BCy, 'x1': x1},
        with_intercept = True,
        verbosity = 0,
        significance_level = 0.05, 
        independence_constant_variance_zero_mean_and_normal_distritbuion_assumptions_met = True)

    #BCfmodel = sm.OLS(BCy, x1).fit()
    # This statement is used by Dr. Yu. My understanding is that both statements should be fine.

    SSE_lambda = ((BCfmodel.predict() - BCy) ** 2).sum()

    loglf = (lambda1 - 1) * ((np.log(y1)).sum()) - len(y1) * (np.log(SSE_lambda / len(y1))) / 2

    print('NOTE: To get the correct prediction, it is almost certain that lec6_box_cox_transform_inverser is needed.')
    return { 'lambda': lambda1, 'loglf': loglf }

def lec6_box_cox_transform_inverser(yt, lambda1=0):
    # http://www.css.cornell.edu/faculty/dgr2/_static/files/R_html/Transformations.html#3_transformation_and_back-transformation

    assert lambda1 != 0 # Not implemented
    return math.exp(np.log(1 + lambda1 * yt) / lambda1)

def lec6_autocorrelation_corrector(
            dataset_with_autocorrelation_issues: pd.DataFrame,
            durbin_watson_stat: float,
            column_names_of_explanatory_aka_independent_variables: List[str],
            column_name_of_response_aka_dependent_variable: str):
    
    ds_old = dataset_with_autocorrelation_issues
    assert isinstance(ds_old, pd.core.frame.DataFrame)
    dw =  durbin_watson_stat
    assert dw <= 4 and dw >= 0
    
    x_names = column_names_of_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable
    
    p = 1 - dw / 2

    ds_new = pd.DataFrame()
    ds_new[y_name] = ds_old[y_name] - p * ds_old[y_name].shift(1)
    for i in range(len(x_names)):
        ds_new[x_names[i]] = ds_old[x_names[i]] - p * ds_old[x_names[i]].shift(1)

    ds_new = ds_new[1:]
    return ds_new

def lec6_ridge_regression(patsy_formula: str,
    variable_names_and_values: Dict[str, np.ndarray],
    name_of_DEpendent_variable: str,
    with_intercept: bool,
    significance_level = 0.05):
    
    print('Ridge regression results:')
    if with_intercept == False:
        print('NOTE: Only scenario with intercept is verified!')
    
    ols_model = lec4_ols_model_helper(
        patsy_formula = patsy_formula,
        variable_names_and_values = variable_names_and_values,
        with_intercept = with_intercept,
        verbosity = 0,
        significance_level = significance_level, 
        independence_constant_variance_zero_mean_and_normal_distritbuion_assumptions_met = True)
    
    if with_intercept:
        k = len(ols_model.params) - 1
    else:
        k = len(ols_model.params)
        
    B = ols_model.params
    s = ols_model.mse_resid ** (0.5)
    hk_lambda = (k + 1) * (s ** 2) / B.dot(B)
    print(f'hk_lambda: {hk_lambda}')
    ridge = Ridge(fit_intercept = False, alpha = hk_lambda)
    # According to Dr. Yu's demonstration, this False is hard-coded even if an intercept is included.
    
    X = pd.DataFrame()
    col_names = ['constant'] if with_intercept else []

    for i, k in enumerate(variable_names_and_values):
        if k != name_of_DEpendent_variable:
            X[k] = variable_names_and_values[k]
            col_names.append(k)
            
    xx = sm.add_constant(X) if with_intercept else X
    xx.columns = col_names
    
    ridge.fit(xx, variable_names_and_values[name_of_DEpendent_variable])
    
    for idx, col_name in enumerate(xx.columns):
        print('The coefficient for {} is {}'.format(col_name, ridge.coef_[idx]))
    
    