import math
from scipy.stats import norm

CRITICAL_THRESHOLDS = {
    0.15: 1.61,
    0.1: 1.933,
    0.05: 2.492,
    0.025: 3.070,
    0.01: 3.857
}

def anderson_darling(
    sample: list[float], 
    mu: float, 
    sigma: float, 
    alpha: float) -> bool:
    """Determines whether Anderson-Darling normality test with known mean, std rejects at given significance level"""
    
    if not(alpha in CRITICAL_THRESHOLDS.keys()):
        raise ValueError("Alpha must be one of 0.01, 0.025, 0.05, 0.1, or 0.15")

    n = len(sample)
    if n < 5:
        raise ValueError("Anderson-Darling test requires at least five samples")
    
    z_scores_sorted = list(map(lambda x: (x-mu)/sigma, sorted(sample)))
    summation = 0
    for i in range(1, n + 1):
        summation += (2*i - 1)*(math.log(norm.cdf(z_scores_sorted[i-1])) + math.log(1-norm.cdf(z_scores_sorted[n-i])))
    
    a_sq = -1*n - summation/n
    return a_sq > CRITICAL_THRESHOLDS[alpha]

