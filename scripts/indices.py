import numpy as np
import pandas as pd
import numpy.typing as npt
import subprocess
import config
from pathlib import Path


def normalized_difference_2band(
    band_a: npt.ArrayLike,
    band_b: npt.ArrayLike,
    error_term: float = config.DENOMINATOR_ERROR_TERM
) -> npt.NDArray:
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
) -> npt.NDArray:
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
    band_names: list[str],
    impute: bool = False,
) -> pd.DataFrame:
    """
    Removes/imputes rows from a DataFrame if the values for ALL bands are outliers
    (i.e., outside the interquartile range for a given band).

    Args:
        data: The input DataFrame.
        band_names: The list of column names to remove outliers.
        impute: If True, replaces outliers with the median value.
            If left False, will simply remove outliers.
    
    Returns:
        Another DataFrame with the outliers removed.
    """

    df = data.copy(deep=True)
    # mask = pd.Series(True, index=df.index)

    for band in band_names:
        q1, q3 = df[band].quantile(0.25), df[band].quantile(0.75)
        median = df[band].median()
        iqr = q3 - q1
        mask = (df[band] < q1 - (1.5 * iqr)) | (df[band] > q3 + (1.5 * iqr))
        print(f'Number of outliers detected for band {band}: {mask.sum()}')
        if impute:
            df.loc[mask, band] = median
        else:
            df = df[~mask]

    # q1 = df[band_names].quantile(0.25)
    # q3 = df[band_names].quantile(0.75)
    # median = df[band_names].quantile(0.5)
    # iqr = q3 - q1

    # mask = ((df[band_names] >= (q1 - (1.5 * iqr))) | (df[band_names] <= (q3 + (1.5 * iqr)))).all(axis=1)
    # print('Number of outliers detected: ', len(df) - np.sum(mask))

    if impute == False:
        return df.reset_index(drop=True)
    else:
        return df


def spectral_feature_polynomial(
    fp_data: str | Path,
    category_column: str,
    target_category: str,
    band_names: list[str],
    fp_script: str | Path = Path(config.FP_SPECTRAL_FEATURE_POLYNOMIAL)
):
    """
    Runs the Spectral Feature Polynomial tool, which takes a multispectral
    dataset in CSV form and discovers a single algebraic index that best
    separates a target land cover class from everything else.

    Specifically, this function uses the `sfp_satellite.py` script, which
    limits the search to illumination-invariant families (2-band normalized
    difference (ND); 3-band ND, and normalized curve).

    The resulting index is a compact algebraic function of a few bands. It
    is easily interpretable and deployable on any remote-sensing plantform,
    and requires no standardization/normalization statistics before input.

    Source: Lotfi, A., Carter, A., Ha, T., Meysami, M., Nketia, K., & 
    Shirtliffe, S. (2026). Interpretable Machine Learning–Derived Spectral 
    Indices for Vegetation Monitoring. Machine Learning with Applications.

    Args:
        fp_data: Filepath to the input CSV with labeled multispectral data
        category_column: Column name for categories (e.g., land cover)
        target_category: Category for which an index is desired
        band_names: List of band column names to include in the search
        fp_script: Filepath to the `sfp_satellite.py` script. Set by default
            to the filepath configured inside `config.py`.
    """
    # Set up the args for the Python script
    args_list = [
        '--csvs', fp_data,
        '--label-col', f'{category_column}',
        '--target', f'{target_category}',
        '--bands', ','.join(band_names)
    ]

    # Set up the Python executable matching that of the tool
    if isinstance(fp_script, str):
        fp_script = Path(fp_script)
    fp_python = fp_script.parent / '.venv' / 'Scripts' / 'python.exe'
    assert fp_python.exists(), f'Filepath to Python executable does not exist: {fp_python}'

    # Build the full command
    command = [fp_python, fp_script] + args_list

    try:
        # Run the command-line tool
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True
        ).stdout
        # return result.stdout
    except subprocess.CalledProcessError as e:
        print(f'SFP tool failed with error code {e.returncode}')
        raise e

    # Extract relevant results and put them in a dictionary
    result_list = result.splitlines()

    best_index = result_list[17].split()[1]
    _, _, n_folds, mean_accuracy, median_accuracy, min_accuracy = tuple(
        result_list[21].split()
    )

    return {
        'best_index': best_index,
        'n_folds': n_folds,
        'mean_accuracy': mean_accuracy,
        'median_accuracy': median_accuracy,
        'min_accuracy': min_accuracy
    }
