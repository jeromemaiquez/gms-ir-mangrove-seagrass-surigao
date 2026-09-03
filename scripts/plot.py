import seaborn as sns
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Ellipse


def _subset_categories(
    data: pd.DataFrame,
    category_column: str,
    cats_compared: list[str] | None = None,
) -> pd.DataFrame:
    """Filter DataFrame rows only to target categories."""

    data_plot = data
    if cats_compared is not None:
        data_plot = data[data[category_column].isin(cats_compared)]

    return data_plot


def spectral_signature(
    data: pd.DataFrame,
    category_column: str,
    value_column: str,
    band_column: str,
    cat_palette: str | dict | None = None,
    cats_compared: list[str] | None = None,
    show_errorbar: bool = True,
    figsize: tuple[int,int] | None = None,
) -> Axes:
    """
    Creates a pointplot showing the spectral signature of different
    categories (e.g., land cover). Can also be used for non-spectral
    variables with ordinal names (e.g., principal components). 

    Ensure that input DataFrame is in long format or 'melted'. See this
    pandas user guide on melting DataFrames for more info: 
    https://pandas.pydata.org/docs/user_guide/reshaping.html#melt-and-wide-to-long
    
    Best used to visualize the "shape" of different classes; since the
    y-axis is shared for all value variables (e.g., bands), those with
    smaller differences may not be easily seen. In those cases, it may
    be better to use the `boxplot_bands()` function.  

    Args:
        data: pd.DataFrame containing the values to aggregate & plot
        category_column: Column name for the categories (e.g., land cover)
        value_column: Column name for the boxplot values
        band_column: Column name for the x-axis categories (e.g., bands)
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
    data_plot = _subset_categories(data, category_column, cats_compared)

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

    Ensure that input DataFrame is in long format or 'melted'. See this
    pandas user guide on melting DataFrames for more info: 
    https://pandas.pydata.org/docs/user_guide/reshaping.html#melt-and-wide-to-long

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
    data_plot = _subset_categories(data, category_column, cats_compared)

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


def _confidence_ellipse(
    x: npt.ArrayLike, 
    y: npt.ArrayLike, 
    ax: Axes, 
    n_std: float = 3.0,
    facecolor: str = 'none',
    **kwargs
) -> Patch:
    """
    Plots the covariance confidence ellipse of `x` and `y`
    on an existing matplotlib Axes.

    Args:
        x, y: Input data
        ax: Axes object to draw the ellipse into
        n_std: Number of standard deviations to determine
            the radius of the ellipse.
        **kwargs: Forwarded to `matplotlib.patches.Ellipse`.

    Returns:
        The Ellipse object (of type Patch) added to `ax`.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def scatterplot_bands(
    data: pd.DataFrame,
    band_x: str,
    band_y: str,
    category_column: str,
    cat_palette: str | dict | None = None,
    cats_compared: list[str] | None = None,
    confidence_ellipse: bool = True,
    n_std: float = 3.0
) -> Axes:
    """
    Creates a two-band scatterplot for visually assessing the
    separability of different classes along two dimensions. Can also
    be used for non-spectral bands (e.g., indices, principal components).

    Also provides the option to add confidence ellipses onto the scatterplot
    to better visualize class overlap and separability. Arg `n_std` refers to
    the number of standard deviations that controls the ellipse radius.

    Ensure that the input DataFrame is in wide format and NOT 'melted'.
    See this pandas user guide on melting DataFrames for more info: 
    https://pandas.pydata.org/docs/user_guide/reshaping.html#melt-and-wide-to-long

    Args:
        data: pd.DataFrame containing the values to plot
        band_x, band_y: The variables to be plotted along the x- and y-axes
        category_column: Column name for the categories shown by hue
        cat_palette: Colors to use for the different categories. Either a
            dict of category:color pairs or a matplotlib/seaborn colormap.
            If left None, default seaborn colormap will be used.
        cats_compared: List of categories to include in the boxplot.
            If left None, all categories will be included.
        show_errorbar: If True, shows the standard deviation of all
            values for a given "band" as the errorbar. Hidden if False.
    """
    data_plot = _subset_categories(data, category_column, cats_compared)

    ax = sns.scatterplot(
        data=data_plot,
        x=band_x,
        y=band_y,
        hue=category_column,
        palette=cat_palette
    )

    if confidence_ellipse == True:    
        unique_cats = data_plot[category_column].unique()
        for i, cat in enumerate(unique_cats):
            # Edge color logic based on input cat_palette
            try:
                edgecolor = cat_palette[cat] # type: ignore
            except TypeError:
                if cat_palette is None:
                    cmap = colormaps["tab10"]
                elif isinstance(cat_palette, str):
                    cmap = colormaps[cat_palette]

                colors = cmap(np.linspace(0, 1, len(unique_cats)))
                edgecolor = colors[i]

            # Generate and add confidence ellipse per category
            data_plot_cat = data_plot[data_plot[category_column] == cat]
            _confidence_ellipse(
                data_plot_cat[band_x], 
                data_plot_cat[band_y],
                ax=ax,
                n_std=n_std,
                edgecolor=edgecolor,
                linewidth=2
            )

    return ax