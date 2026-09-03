import numpy as np
import numpy.typing as npt
import numexpr as ne
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