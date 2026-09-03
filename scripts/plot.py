import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

def spectral_signature(
    data: pd.DataFrame,
    category_column: str,
    value_column: str,
    band_column: str,
    cat_palette: str | dict | None = None,
    cats_compared: list[str] | None = None,
    show_errorbar: bool = True
) -> Axes:
    """
    Creates a pointplot showing the spectral signature of different
    categories (e.g., land cover). Can also be used for non-spectral
    variables with ordinal names (e.g., principal components). 
    
    Best used to visualize the "shape" of different classes; since the
    y-axis is shared for all value variables (e.g., bands), those with
    smaller differences may not be easily seen. In those cases, it may
    be better to use the `boxplot_bands()` function.  

    Args:
        data: pd.DataFrame containing the values to aggregate & plot
        category_column:Column name for the boxplot categories
        value_column: Column name for the boxplot values
        band_column: Column name determining the faceting of the grid
        cat_palette: Colors to use for the different categories. Either a
            dict of category:color pairs or a matplotlib/seaborn colormap.
            If left None, default seaborn colormap will be used.
        cats_compared: List of categories to include in the boxplot.
            If left None, all categories will be included.
        show_errorbar: If True, shows the standard deviation of all
            values for a given "band" as the errorbar. Hidden if False.

    Returns:
        A matplotlib Axes of the pointplot.
    """
    data_plot = data
    if cats_compared is not None:
        data_plot = data[data[category_column].isin(cats_compared)]

    errorbar = 'sd'
    if show_errorbar == False:
        errorbar = None

    ax = sns.pointplot(
        data=data_plot,
        y=value_column,
        x=band_column,
        hue=category_column,
        palette=cat_palette,
        errorbar=errorbar,
        dodge=True
    )

    return ax


def boxplot_bands(
    data: pd.DataFrame,
    category_column: str,
    value_column: str,
    band_column: str,
    cat_palette: str | dict | None = None,
    cats_compared: list[str] | None = None,
    col_wrap: int | None = None,
    sharey: bool = False
) -> sns.FacetGrid:
    """
    Creates a FacetGrid of boxplots where axes represent different 
    bands/indices. Useful to visually assess class separability.
    Categories are on the x-axis, and values on the y-axis.

    Arg `band_column` can also be used for things other than spectral
    bands, such as different indices or principal components.

    Args:
        data: pd.DataFrame containing the values to aggregate & plot
        category_column:Column name for the boxplot categories
        value_column: Column name for the boxplot values
        band_column: Column name determining the faceting of the grid
        cat_palette: Colors to use for the different categories. Either a
            dict of category:color pairs or a matplotlib/seaborn colormap.
            If left None, default seaborn colormap will be used.
        cats_compared: List of categories to include in the boxplot.
            If left None, all categories will be included.
        col_wrap: "Wrap" the column variable (i.e. `band_column`) at
            this width, so that the facets span multiple rows.
            If left None, auto-calculated based on number of categories.
        sharey: If True, facets will share y-axes (i.e. value ranges).
            False by default to show different ranges per band/index.

    Returns:
        A seaborn FacetGrid of boxplots per unique `band_column` value.
    """
    data_plot = data
    if cats_compared is not None:
        data_plot = data[data[category_column].isin(cats_compared)]

    g = sns.catplot(
        kind='box',
        data=data_plot,
        y=value_column,
        x=category_column,
        hue=category_column,
        col=band_column,
        palette=cat_palette,
        col_wrap=col_wrap,
        sharey=sharey
    )

    return g