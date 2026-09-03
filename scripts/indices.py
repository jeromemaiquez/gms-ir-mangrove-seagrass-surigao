import numpy as np
import pandas as pd
import numpy.typing as npt
import config


def normalized_difference_2band(
    band_a: npt.ArrayLike,
    band_b: npt.ArrayLike,
    error_term: float = config.DENOMINATOR_ERROR_TERM
) -> npt.ArrayLike:
    """
    Calculates the normalized difference between two arrays `band_a` and
    `band_b` element-wise. The arrays must have the same shape.

    Args:
        band_a, band_b: Input arrays with the same shape
        error_term: Error term to prevent zero division.

    Returns:
        Array of normalized difference values with same shape as inputs.
    """
    band_a = np.asarray(band_a).astype(float)
    band_b = np.asarray(band_b).astype(float)

    if band_a.shape != band_b.shape:
        raise ValueError("Band value arrays must have the same length/shape.")

    return (band_a - band_b) / (band_a + band_b + error_term)


def normalized_difference_3band(
    band_a: npt.ArrayLike,
    band_b: npt.ArrayLike,
    band_c: npt.ArrayLike,
    which_negative: str,
    error_term: float = config.DENOMINATOR_ERROR_TERM
) -> npt.ArrayLike:
    """
    Calculates the normalized difference between three arrays `band_a`, `band_b`, 
    and `band_c` (all with the same shape), following the general formula:
    
    (s_a*band_a + s_b*band_b + s_c*band_c) / (band_a + band_b + band_c)

    where (s_a, s_b, s_c) are sign patterns with exactly one negative term.
    The arg `which_negative` controls which term is negative.

    Args:
        band_a, band_b, band_c: Input arrays with the same shape
        error_term: Error term to prevent zero division.

    Returns:
        Array of normalized difference values with same shape as inputs.    
    """
    # Assign which term is negative
    match which_negative:
        case "a"|"A":
            s_a, s_b, s_c = (-1, 1, 1)
        case "b"|"B":
            s_a, s_b, s_c = (1, -1, 1)
        case "c"|"C":
            s_a, s_b, s_c = (1, 1, -1)
        case _:
              raise ValueError("Only one of bands A, B, or C can be negative.")

    band_a = np.asarray(band_a).astype(float)
    band_b = np.asarray(band_b).astype(float)
    band_c = np.asarray(band_c).astype(float)

    if band_a.shape != band_b.shape != band_c.shape:
            raise ValueError("Band value arrays must have the same length/shape.")

    return ((s_a * band_a) + (s_b * band_b) + (s_c + band_c)) / (band_a + band_b + band_c + error_term) 


def remove_outliers(
    data: pd.DataFrame,
    band_names: list[str]
) -> pd.DataFrame:
    """
    Removes rows from a DataFrame if the values for ALL bands are outliers
    (i.e., outside the interquartile range for a given band).

    Args:
        data: The input DataFrame.
        band_names: The list of column names to remove outliers.
    
    Returns:
        Another DataFrame with the outliers removed.
    """

    df = data.copy(deep=True)

    q1 = df[band_names].quantile(0.25)
    q3 = df[band_names].quantile(0.75)
    iqr = q3 - q1

    mask = ((df[band_names] >= (q1 - (1.5 * iqr))) | (df[band_names] <= (q3 + (1.5 * iqr)))).all(axis=1)
    print('Number of outliers detected: ', np.sum(mask))
    return df[mask].reset_index(drop=True)