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

# --------------
# Initial Set-up 
# --------------

# Set up DuckDB connection and spatial extensions
con = duckdb.connect()
con.execute('INSTALL spatial; LOAD spatial;')

# Filepaths for ADM4 boundaries and S2Coast datasets
fp_adm4 = Path(r"C:\\Users\\remot\Documents\\Jerome\\05_GIS_Data\\Vector Data\\Edge-Matched Global Subnational Boundaries\\adm4_polygons.parquet")
fp_s2coast = Path(r"C:\\Users\\remot\\Documents\\Jerome\\05_GIS_Data\\Vector Data\\S2Coast\\S2Coast2023_ShapeFile_vector\\S2Coast-2023_Polyline_diss.shp")
fp_s2coast_tiles = Path(r"C:\\Users\\remot\\Documents\\Jerome\\05_GIS_Data\\Vector Data\\S2Coast\\SupportVectorFiles\\Fishnet_1Dedgree.shp")

# URL for FABDEM STAC catalog
fabdem_catalog_url = 'https://huggingface.co/datasets/links-ads/fabdem-v12/raw/main/stac_catalog/catalog.json'

# Set province and municipality name for AOI
province = 'Surigao del Sur'
municipality = 'Hinatuan'

# ----------------------
# Preparing input layers
# ----------------------

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

# Define CRSs for future buffering and matching
crs_wgs84 = gdf_admin_bounds.crs
crs_utm50n = 'EPSG:32651'

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
# print(gdf_neighbors.head())

# Create ocean mask for future clipping of offshore portion of AOI
gs_ocean_mask = (
    gdf_admin_bounds
        .to_crs(crs_utm50n)
        .buffer(5_000)
        .to_crs(crs_wgs84)
        .difference(gdf_admin_bounds)
        .difference(gdf_neighbors.union_all())
)

# Query the S2Coast tile intersecting with the AOI
query_coast = f"""
SELECT *
FROM '{str(fp_s2coast)}'
WHERE Location IN (
    SELECT t.Location
    FROM '{str(fp_s2coast_tiles)}' AS t
    JOIN arrow_admin_bounds AS a ON ST_Intersects(
        t.geom, 
        a.geometry
    )
)
"""

gdf_coast = gpd.GeoDataFrame.from_arrow(
    con.sql(query_coast).arrow()
).set_crs(crs_wgs84)
# print(gdf_coast)

# Download the OSM boundary (which contains both and land and sea territory) for future clipping
gdf_osm_aoi = ox.geocode_to_gdf(f'{municipality}, {province}, Philippines')

# Get coastline clipped to OSM boundaries
gs_coastline = gdf_coast.clip(gdf_osm_aoi.to_crs(gdf_coast.crs)).geometry

# ------------------------
# Generate portions of AOI
# ------------------------

# Portion 1: 1-km inland buffer
gs_inland = gpd.GeoSeries(
    gs_coastline
        .to_crs(crs_utm50n)
        .buffer(1_000)
        .to_crs(crs_wgs84)
        .clip(gdf_admin_bounds)
        .union_all(),
    crs=crs_wgs84
)

# Portion 2: 2-km offshore buffer
gs_offshore = gpd.GeoSeries(
    gs_coastline
        .to_crs(crs_utm50n)
        .buffer(2_000)
        .to_crs(crs_wgs84)
        .clip(gs_ocean_mask)
        .union_all(),
    crs=crs_wgs84
).clip(gdf_osm_aoi.to_crs(crs_wgs84))