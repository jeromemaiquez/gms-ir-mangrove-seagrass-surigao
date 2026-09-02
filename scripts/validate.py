import numpy as np
from numpy.typing import ArrayLike

def m_statistic(
    mean_a: int | float,
    mean_b: int | float,
    std_a: int | float,
    std_b: int | float
) -> float:
    """
    Calculates the M statistic as a measure of separability between two 
    classes. The difference between two class means is normalized by the 
    sum of their standard deviations. M-statistic values greater than 1 
    indicate good separability, while values under 1 indicate
    large overlaps between the two class histograms.

    M = |mean_a - mean_b| / (std_a + std_b)

    Source: Kaufman, Y.J. and Remer, L., 1994. Remote sensing of 
    vegetation in the mid-IR: The 3.75 µm channels. IEEE Trans. 
    Geosci. Remote Sens, 32, pp.672-683.

    Parameters
    ----------
    mean_a, mean_b: int | float
        Mean values of class A and B
    std_a, std_b: int | float
        Standard deviations of class A and B.

    Returns
    -------
    float
        M-statistic as a measure of separability.
    """

    return np.abs(mean_a - mean_b) / (std_a + std_a)

def _bhattacharyya(
    mean_a: int | float,
    mean_b: int | float,
    std_a: int | float,
    std_b: int | float 
) -> float:
    """
    Calculates the Bhattacharyya distance between two classes.
    
    Source: Bhattacharya, A., 1946. On a measure of divergence
    between two multinomial populations. Sankhya: the Indian
    Journal of Statistics., pp.401-406.
    """

    return (
        0.25 * np.log(0.25 * ((std_a**2 / std_b**2) + (std_b**2 / std_a**2) + 2))
        + (0.25 * ((mean_a - mean_b)**2 / (std_a**2 + std_b**2)))
    )

def jm_distance(
    mean_a: int | float,
    mean_b: int | float,
    std_a: int | float,
    std_b: int | float 
) -> float:
    """
    Calculates the Jeffries-Matusita distance between two classes,
    which standardizes the Bhattacharyya distance to a range of 0-2.

    Source: Sen, R., Goswami, S., and Chakraborty, B., 2019. Jeffries-
    Matusita distance as a tool for feature selection.
    """
    bhattacharyya = _bhattacharyya(mean_a, mean_b, std_a, std_b)
    return np.sqrt(2 * (1 - np.exp(-bhattacharyya)))