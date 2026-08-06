import duckdb
import geopandas as gpd
import pandas as pd
import osmnx as ox
import rioxarray as rxr

from shapely import to_geojson
from shapely.geometry import LineString
from rasterio.features import shapes

from fabdem import load_filtered_items

import pystac
import pystac_client
from odc.stac import load
from pathlib import Path

# Set up DuckDB connection and spatial extensions
con = duckdb.connect()
con.execute('INSTALL spatial; LOAD spatial;')

# Filepaths for ADM4 boundaries and S2Coast datasets
fp_adm4 = Path(r"C:\\Users\\remot\Documents\\Jerome\\05_GIS_Data\\Vector Data\\Edge-Matched Global Subnational Boundaries\\adm4_polygons.parquet")
fp_s2coast = Path(r"C:\\Users\\remot\\Documents\\Jerome\\05_GIS_Data\\Vector Data\\S2Coast\\S2Coast2023_ShapeFile_vector\\S2Coast-2023_Polyline_diss.shp")

# URL for FABDEM STAC catalog
fabdem_catalog_url = 'https://huggingface.co/datasets/links-ads/fabdem-v12/raw/main/stac_catalog/catalog.json'

# Set province and municipality name for AOI
province = 'Surigao del Sur'
municipality = 'Hinatuan'

# Query boundaries of target municipality
query_admin_bounds = f"""
SELECT *
FROM '{str(fp_adm4)}'
WHERE
    adm0_name = 'Philippines'       AND
    adm2_name = '{province}'        AND
    adm3_name = '{municipality}'
"""

gdf_admin_bounds = gpd.GeoDataFrame.from_arrow(
    con.sql(query_admin_bounds).arrow()
)
# print(gdf_admin_bounds.head())

# Convert admin bounds GeoDataFrame back to GeoArrow for future DuckDB queries
arrow_admin_bounds = gdf_admin_bounds.to_arrow()

# Query neighbors of target municipality (to ensure AOI only falls inside target)
query_neighbors = f"""
SELECT adm4_polygons.*
FROM '{str(fp_adm4)}'
JOIN arrow_admin_bounds ON ST_Intersects(
    adm4_polygons.geometry,
    arrow_admin_bounds.geometry
)
WHERE
    adm4_polygons.adm3_name != '{municipality}'
"""

gdf_neighbors = gpd.GeoDataFrame.from_arrow(
    con.sql(query_neighbors).arrow()
)
print(gdf_neighbors.head())  