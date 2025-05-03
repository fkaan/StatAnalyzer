import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def t_test(data, col1, col2):
    """
    Perform independent samples t-test between two metric variables
    Returns t-statistic and p-value
    """
    t_stat, p_val = stats.ttest_ind(data[col1], data[col2], nan_policy='omit')
    return t_stat, p_val

def anova_test(data, metric_col, group_col):
    """
    Perform one-way ANOVA test
    Returns F-statistic and p-value
    """
    groups = data.groupby(group_col)[metric_col].apply(list)
    f_stat, p_val = stats.f_oneway(*groups)
    return f_stat, p_val

def chi_square_test(data, col1, col2):
    """
    Perform chi-square test of independence between two categorical variables
    Returns chi-square statistic and p-value
    """
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency_table)
    return chi2_stat, p_val

def tukey_hsd(data, metric_col, group_col):
    """
    Perform Tukey's HSD post-hoc test for ANOVA
    Returns formatted results string
    """
    tukey = pairwise_tukeyhsd(endog=data[metric_col],
                             groups=data[group_col],
                             alpha=0.05)
    result = str(tukey.summary())
    return result

def welch_anova(data, metric_col, group_col):
    """
    Perform Welch's ANOVA for unequal variances
    Returns F-statistic and p-value
    """
    groups = data.groupby(group_col)[metric_col].apply(list)
    f_stat, p_val = stats.f_oneway(*groups)
    return f_stat, p_val

def mann_whitney_u(data, col1, col2):
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test)
    Returns U-statistic and p-value
    """
    u_stat, p_val = stats.mannwhitneyu(data[col1], data[col2])
    return u_stat, p_val

def kruskal_wallis(data, metric_col, group_col):
    """
    Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
    Returns H-statistic and p-value
    """
    groups = data.groupby(group_col)[metric_col].apply(list)
    h_stat, p_val = stats.kruskal(*groups)
    return h_stat, p_val