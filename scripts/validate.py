import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def agg_per_class(
    data: pd.DataFrame,
    category_column: str,
    band_names: list[str],
    cats_compared: list[str]
) -> pd.DataFrame:
    """
    Aggregates the band values into per-class means and standard deviations.
    Preparatory step for separability metric calculations.
    
    Args:
        data: Input pandas DataFrame to aggregate per category
        category_column: Column name for category (e.g., land cover)
        band_names: List of column names corresponding to bands/indices
        cats_compared: Pair of categories to compare for separability calculation
    
    Returns:
        A new wide-format DataFrame with means and standard deviations per class.
        Stat columns are in format '{band}_{mean/std}' 
    """
    data = data.copy(deep=True)
    data = data[data[category_column].isin(cats_compared)]
    if len(data) == 0:
        raise ValueError("DataFrame is empty, expected at least 1 row.")

    df_stat = data.groupby(category_column)[band_names].agg(['mean', 'std']).reset_index()

    # Collapse multi-level columns
    stat_names = ['_'.join(col).strip() for col in list(df_stat.columns.values)[1:]]
    df_stat.columns = [category_column] + stat_names

    # # Melt stat DataFrame for easier querying later on
    # df_stat_melt = df_stat.melt(
    #     id_vars=category_column,
    #     value_vars=stat_names,
    #     value_name='value',
    #     var_name='column'
    # )

    # # Split column name into band & stat names
    # df_stat_melt[['band', 'stat']] = df_stat_melt['column'].str.split("_", expand=True)
    # return df_stat_melt.drop(columns=['column'])

    return df_stat


def get_separability(
    df_stats: pd.DataFrame,
    category_column: str,
    category_a: str,
    category_b: str,
    band_names: list[str],
) -> pd.DataFrame:
    """
    Creates a new DataFrame with the separability metrics for
    `category_a` vs. `category_b` per band in `band_names`.
    Batch calculation of separability metrics for multiple bands/indices.

    Args:
        df_stats: Input DataFrame of band means & stds per class
        category_column: Column name for category (e.g., land cover)
        category_a, category_b: Classes to assess for separability
        band_names: List of bands/indices for which to calculate separability

    Returns:
        New DataFrame with separability metrics per band/index.
    """
    separability_per_band = []

    for band in band_names:
        mean_a, std_a = df_stats.loc[df_stats[category_column] == category_a, [f'{band}_mean', f'{band}_std']].values.flatten()
        mean_b, std_b = df_stats.loc[df_stats[category_column] == category_b, [f'{band}_mean', f'{band}_std']].values.flatten()

        m_stat = m_statistic(mean_a, mean_b, std_a, std_b)
        jm_dist = jm_distance(mean_a, mean_b, std_a, std_b)

        separability_per_band.append({
            'band': band,
            'm_statistic': m_stat,
            'jm_distance': jm_dist
        })

    return pd.DataFrame(separability_per_band)


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

    Args:
        mean_a, mean_b: Mean values of class A and B
        std_a, std_b: Standard deviations of class A and B.

    Returns
        M-statistic as a measure of separability.
    """
    if (std_a == 0) and (std_b == 0):
        raise ValueError("Standard deviations cannot both be zero.")
    
    return np.abs(mean_a - mean_b) / (std_a + std_b)

def _bhattacharyya(
    mean_a: int | float,
    mean_b: int | float,
    std_a: int | float,
    std_b: int | float 
) -> float:
    """
    Calculates the Bhattacharyya distance between two classes.

    D_B = 1/4 * ln(1/4 * (std_a^2 / std_b^2) + (std_b^2 / std_a^2))
    + (1/4 * (mean_a - mean_b)^2 / (std_a^2 + std_b^2))
    
    Formula source: https://medium.com/@yoavyeledteva/bhattacharyya
    -distance-from-statistics-to-application-in-data-science-8eb5ccdbba62
    
    Source: Bhattacharya, A., 1946. On a measure of divergence
    between two multinomial populations. Sankhya: the Indian
    Journal of Statistics., pp.401-406.

    Args:
        mean_a, mean_b: Mean values of class A and B
        std_a, std_b: Standard deviations of class A and B.

    Returns
        M-statistic as a measure of separability.
    """
    if (std_a == 0) and (std_b == 0):
        raise ValueError("Standard deviations cannot both be zero.")
    
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

    Args:
        mean_a, mean_b: Mean values of class A and B
        std_a, std_b: Standard deviations of class A and B.

    Returns
        M-statistic as a measure of separability.
    """
    bhattacharyya = _bhattacharyya(mean_a, mean_b, std_a, std_b)
    return np.sqrt(2 * (1 - np.exp(-bhattacharyya)))