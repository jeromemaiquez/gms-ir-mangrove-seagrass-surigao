import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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