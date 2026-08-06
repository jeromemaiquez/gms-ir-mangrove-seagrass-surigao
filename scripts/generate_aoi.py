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