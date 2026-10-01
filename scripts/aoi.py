import duckdb
import geopandas as gpd
import pandas as pd
import osmnx as ox
import rioxarray as rxr

from shapely import to_geojson
from shapely.geometry import LineString, Polygon, MultiPolygon
from rasterio.features import shapes
import matplotlib.pyplot as plt

from fabdem import load_filtered_items

import pystac
import pystac_client
from odc.stac import load
from pathlib import Path


def get_municipality_bounds(
    con: duckdb.DuckDBPyConnection,
    fp_bounds: str | Path,
    province: str,
    muni_names: list[str],
    prov_col: str,
    muni_col: str
):
    """
    Returns a GeoDataFrame of the target municipalities queried
    from an administrative boundaries dataset via DuckDB.

    Args:
        con: Active DuckDB connection
        fp_bounds: Filepath to the administrative boundaries dataset
        province: Name of the province where the municipalities are
        muni_names: List of city/municipality names
        prov_col: Column/field name for provinces
        muni_col: Column/field name for cities/municipalities
    
    Returns:
        A geopandas.GeoDataFrame of the results.

    Raises:
        KeyError if no municipalities match province and muni_names
    """
    muni_names_sql = ','.join([f"'{muni}'" for muni in muni_names])

    query_municipality_bounds = f"""
    SELECT *
    FROM '{str(fp_bounds)}'
    WHERE
        {prov_col} = '{province}'        AND
        {muni_col} IN ({muni_names_sql})
    """

    gdf_municipality_bounds = gpd.GeoDataFrame.from_arrow(
        con.sql(query_municipality_bounds).arrow()
    ).dissolve()

    if len(gdf_municipality_bounds) == 0:
        raise KeyError("No municipalities matched the provided province and muni_names.")

    return gdf_municipality_bounds


def get_neighbor_bounds(
    con: duckdb.DuckDBPyConnection,
    fp_bounds: str | Path,
    gdf_municipality: gpd.GeoDataFrame,
    muni_names: list[str],
    muni_col: str
):
    """
    Returns a GeoDataFrame of the neighbors surrounding a target municipality 
    queried from an administrative boundaries dataset via DuckDB.

    Args:
        con: Active DuckDB connection
        fp_bounds: Filepath to the administrative boundaries dataset
        muni_names: List of city/municipality names.
        muni_col: Column/field name for cities/municipalities
    
    Returns:
        A geopandas.GeoDataFrame of the results.
    """

    arrow_municipality = gdf_municipality.to_arrow()
    muni_names_sql = ','.join([f"'{muni}'" for muni in muni_names])
    base_fn_bounds = Path(fp_bounds).stem

    # Discover name of geometry column
    cols = con.execute(f"""
        DESCRIBE SELECT * 
        FROM '{str(fp_bounds)}'
    """).fetchall()
    geom_col = next(row[0] for row in cols if "GEOMETRY" in row[1].upper() or "BLOB" in row[1].upper())

    query_neighbors = f"""
    SELECT {base_fn_bounds}.*
    FROM '{str(fp_bounds)}'
    JOIN arrow_admin_bounds ON ST_Intersects(
        {base_fn_bounds}.{geom_col},
        arrow_admin_bounds.geometry
    )
    WHERE
        {base_fn_bounds}.{muni_col} NOT IN ({muni_names_sql})
    """

    gdf_neighbors = gpd.GeoDataFrame.from_arrow(
        con.sql(query_neighbors).arrow()
    )
    
    if gdf_neighbors.active_geometry_name != 'geometry':
        gdf_neighbors = gdf_neighbors.rename(
            columns={gdf_neighbors.active_geometry_name: 'geometry'}
        ).set_geometry('geometry')

    return gdf_neighbors


def merge_bounds_layers(
    gdf_with_islands: gpd.GeoDataFrame,
    gdf_no_islands: gpd.GeoDataFrame
):
    """
    Returns a GeoDataFrame merging two original administrative boundaries
    datasets, one of which includes small islands. This function is designed
    to harness both the more accurate boundaries of the NAMRIA dataset while
    including the small islands in the Edge-Matched dataset.

    Args:
        gdf_with_islands: GeoDataFrame that includes islands (e.g., Edge-Matched)
        gdf_no_islands: GeoDataFrame that lacks islands (e.g., NAMRIA)
    
    Returns:
        A geopandas.GeoDataFrame of the results.
    """
    gdf_no_islands = gdf_no_islands.to_crs(gdf_with_islands.crs)

    gdf_small_islands = gdf_with_islands.explode()[
        ~gdf_with_islands.explode().intersects(gdf_no_islands.geometry)
    ]

    gdf_merged = gpd.GeoDataFrame(
        pd.concat([gdf_no_islands, gdf_small_islands], ignore_index=True),
        geometry='geometry',
        crs=gdf_with_islands.crs
    ).dissolve()

    return gdf_merged


def extract_coastline(
        gdf_bounds: gpd.GeoDataFrame,
        gdf_neighbors: gpd.GeoDataFrame,
    ):
    """
    Returns a GeoDataFrame with LineString geometry representing the target
    AOI's coastline (i.e., borders that touch the sea). This is done by removing
    all the boundary line segments that border the AOI's neighbors.

    Args:
        gdf_bounds: GeoDataFrame of the target AOI thus far
        gdf_neighbors: GeoDataFrame of the AOI's neighboring municipalities
    
    Returns:
        A geopandas.GeoDataFrame of the results.
    """

    # Explode AOI boundaries into individual segments and remove those bordering neighbors
    gdf_border_segments = gdf_bounds.explode()
    gdf_border_segments['geometry'] = gdf_border_segments.geometry.apply(lambda geom: geom.exterior)
    gdf_border_segments = gdf_border_segments.explode().difference(gdf_neighbors.geometry)

    gdf_coast = gpd.GeoDataFrame(
        data=gdf_bounds,
        geometry=[gdf_border_segments.union_all().simplify(0.0002)],
        crs=gdf_bounds.crs
    )

    return gdf_coast
