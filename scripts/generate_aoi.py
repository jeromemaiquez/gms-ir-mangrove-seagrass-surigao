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


# --- SECTION 0: Initial Set-up ---


# Set up DuckDB connection and spatial extensions
con = duckdb.connect()
con.execute('INSTALL spatial; LOAD spatial;')

# Directories for plots & outputs
work_dir = Path().resolve()
plots_dir = work_dir / 'plots'
output_dir = work_dir / 'output'

# Filepaths for ADM4 boundaries and S2Coast datasets
fp_adm4 = Path(r"C:\\Users\\remot\Documents\\Jerome\\05_GIS_Data\\Vector Data\\Edge-Matched Global Subnational Boundaries\\adm4_polygons.parquet")
fp_namria = Path(r"C:\\Users\\remot\\Documents\\Jerome\\05_GIS_Data\\Vector Data\\PH Admin Bounds\\phl_admbnda_adm3_psa_namria_20231106.shp")
fp_maritime = Path(r"C:\\Users\\remot\\Documents\\Jerome\\05_GIS_Data\\Vector Data\\PH Maritime Boundaries per Municipality\\PH_MaritimeBound_PerMuni_NAMRIA.parquet")
fp_s2coast = Path(r"C:\\Users\\remot\\Documents\\Jerome\\05_GIS_Data\\Vector Data\\S2Coast\\S2Coast2023_ShapeFile_vector\\S2Coast-2023_Polyline_diss.shp")
fp_s2coast_tiles = Path(r"C:\\Users\\remot\\Documents\\Jerome\\05_GIS_Data\\Vector Data\\S2Coast\\SupportVectorFiles\\Fishnet_1Dedgree.shp")

# URL for FABDEM STAC catalog
# fabdem_catalog_url = 'https://huggingface.co/datasets/links-ads/fabdem-v12/raw/main/stac_catalog/catalog.json'
fabdem_catalog_url = 'https://huggingface.co/datasets/links-ads/fabdem-v12/resolve/main/collection.json'

# Set province and municipality name for AOI
province = 'Palawan'
municipality = ['Araceli']


# --- SECTION 1: Preparing input layers ---

# Parse municipality names for SQL
muni_names_sql = ','.join([f"'{muni}'" for muni in municipality])

# Query boundaries of target municipality
query_admin_bounds = f"""
SELECT *
FROM '{str(fp_adm4)}'
WHERE
    adm0_name = 'Philippines'       AND
    adm2_name = '{province}'        AND
    adm3_name IN ({muni_names_sql})
"""

gdf_admin_bounds = gpd.GeoDataFrame.from_arrow(
    con.sql(query_admin_bounds).arrow()
).dissolve()
# print(gdf_admin_bounds.head())

# Define CRSs for future buffering and matching
crs_wgs84 = gdf_admin_bounds.crs
crs_utm50n = 'EPSG:32651'

# Convert admin bounds GeoDataFrame back to GeoArrow for future DuckDB queries
arrow_admin_bounds = gdf_admin_bounds.to_arrow()

# Query boundaries (for NAMRIA dataset)
# con.execute("SET gdal_config_options = 'OGR_SHAPE_NULL_AS_EMPTY=NO';")
query_admin_namria = f"""
SELECT *
FROM ST_Read('{str(fp_namria)}')
WHERE
    ADM2_EN = '{province}'        AND
    ADM3_EN IN ({muni_names_sql})
"""

gdf_admin_namria = gpd.GeoDataFrame.from_arrow(
    con.sql(query_admin_namria).arrow()
).dissolve().rename(columns={'geom': 'geometry'}).set_geometry('geometry')

arrow_admin_namria = gdf_admin_namria.to_arrow()

# Query neighbors of target municipality (to ensure AOI only falls inside target)
query_neighbors = f"""
SELECT adm4_polygons.*
FROM '{str(fp_adm4)}'
JOIN arrow_admin_bounds ON ST_Intersects(
    adm4_polygons.geometry,
    arrow_admin_bounds.geometry
)
WHERE
    adm4_polygons.adm3_name NOT IN ({muni_names_sql})
"""

gdf_neighbors = gpd.GeoDataFrame.from_arrow(
    con.sql(query_neighbors).arrow()
)
# print(gdf_neighbors.head())

query_neighbors_namria = f"""
SELECT phl_admbnda_adm3_psa_namria_20231106.*
FROM ST_Read('{str(fp_namria)}') AS phl_admbnda_adm3_psa_namria_20231106
JOIN arrow_admin_namria ON ST_Intersects(
    phl_admbnda_adm3_psa_namria_20231106.geom,
    arrow_admin_namria.geometry
)
WHERE
    phl_admbnda_adm3_psa_namria_20231106.ADM3_EN NOT IN ({muni_names_sql})
"""

gdf_neighbors_namria = gpd.GeoDataFrame.from_arrow(
    con.sql(query_neighbors_namria).arrow()
).rename(columns={'geom': 'geometry'}).set_geometry('geometry')

# gdf_neighbors_namria.to_file(output_dir / f'{municipality[0]}_Neighbors_NAMRIA.geojson')

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
# query_coast = f"""
# SELECT *
# FROM '{str(fp_s2coast)}'
# WHERE Location IN (
#     SELECT t.Location
#     FROM '{str(fp_s2coast_tiles)}' AS t
#     JOIN arrow_admin_bounds AS a ON ST_Intersects(
#         t.geom, 
#         a.geometry
#     )
# )
# """

# gdf_coast = gpd.GeoDataFrame.from_arrow(
#     con.sql(query_coast).arrow()
# ).set_crs(crs_wgs84)
# print(gdf_coast.head())
# print(gdf_coast)

# Merge edge-matched dataset's small islands with NAMRIA borders and use as AOI
gdf_small_islands = gdf_admin_bounds.explode()[
    ~gdf_admin_bounds.explode().intersects(gdf_admin_namria.to_crs(crs_wgs84))
]
gdf_admin_bounds = gpd.GeoDataFrame(
    pd.concat([gdf_admin_namria.to_crs(crs_wgs84), gdf_small_islands], ignore_index=True),
    geometry='geometry',
    crs=crs_wgs84
).dissolve()

# gdf_admin_bounds.to_file(output_dir / f'{municipality[0]}_AOIwithSmallIslands.geojson')

gdf_border_segments = gdf_admin_bounds.explode()
gdf_border_segments['geometry'] = gdf_border_segments.geometry.apply(lambda geom: geom.exterior)
gdf_border_segments = gdf_border_segments.explode().difference(gdf_neighbors_namria.to_crs(crs_wgs84).union_all())
# gdf_border_segments = gdf_border_segments[
#     (gdf_border_segments.disjoint(gdf_neighbors_namria.to_crs(crs_wgs84).union_all()))
#     | (gdf_border_segments.touches(gdf_neighbors_namria.to_crs(crs_wgs84).union_all()))
# ]

# gdf_border_segments.to_file(output_dir / f'{municipality[0]}_BorderSegments.geojson')

# Extract coastline
gdf_coast = gpd.GeoDataFrame(
    data=gdf_admin_bounds,
    geometry=[gdf_border_segments.union_all().simplify(0.0002)],
    crs=crs_wgs84
)

# print(gdf_coast.head())

# Download the OSM boundary (which contains both and land and sea territory) for future clipping
# osm_muni_queries = []
# for muni in municipality:
#     osm_muni_queries.append(f'{muni}, {province}, Philippines')
# # for muni in [municipality[0]]:
# #     osm_muni_queries.append(f'{muni} Island, {province}, Philippines')

# gdf_osm_aoi = ox.geocode_to_gdf(osm_muni_queries, which_result=1)

query_maritime = f"""
SELECT *
FROM '{str(fp_maritime)}'
WHERE
    ADM2_EN = '{province}'        AND
    ADM3_EN IN ({muni_names_sql})
"""

gdf_maritime = gpd.GeoDataFrame.from_arrow(
    con.sql(query_maritime).arrow()
).dissolve().rename(columns={'geom': 'geometry'}).set_geometry('geometry')

gdf_osm_aoi = gpd.GeoDataFrame(
    pd.concat([gdf_admin_bounds, gdf_maritime.to_crs(crs_wgs84)], ignore_index=True),
    geometry='geometry',
    crs=crs_wgs84
).dissolve()

gdf_osm_aoi.to_file(output_dir / f'{municipality[0]}_CombinedAOIwithHoles.geojson')

# if isinstance(gdf_osm_aoi.geometry.values[0], MultiPolygon):
#     gdf_osm_aoi['geometry'] = MultiPolygon([poly.exterior for poly in gdf_osm_aoi.geometry.values[0].geoms])
# elif isinstance(gdf_osm_aoi.geometry.values[0], Polygon):
#     gdf_osm_aoi['geometry'] = Polygon(gdf_osm_aoi.geometry.values[0].exterior)

# gdf_osm_aoi.to_file(output_dir / f'{municipality[0]}_CombinedAOI.geojson')
# print(gdf_osm_aoi.head())

# Get coastline clipped to OSM boundaries
gs_coastline = gdf_coast.clip(gdf_osm_aoi).geometry
# gs_coastline = gdf_coast.geometry

# --- SECTION 2: Generate inland & offshore portions of AOI ---


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

# Combine to create preliminary AOI
gs_prelim_aoi = gpd.GeoSeries(
    pd.concat([gs_inland, gs_offshore]),
    crs=crs_wgs84
)


# --- SECTION 3: Generate low-elevation mask to add to AOI


# Get bounding box of admin bounds for FABDEM query
bbox = gdf_admin_bounds.total_bounds

# Load filtered FABDEM tiles that intersect with bbox into a DataArray
filtered_items = load_filtered_items(fabdem_catalog_url, tuple(bbox))

da_dem = load(
    filtered_items,
    resolution=30,
    crs="utm"
).squeeze()['DEM']

# Mask to only retain low-elevation pixels
da_low_elev = da_dem.where((da_dem <= 6) & (da_dem >=0))

# Get the DEM's transform and CRS for later use
transform = da_low_elev.rio.transform()
crs_dem = da_low_elev.rio.crs

# Vectorize low-elevation DEM pixels
poly_low_elev = list(
    {
        'geometry': s,
        'properties': {
            'val': v
        }
    } for i, (s, v) in enumerate(shapes(
        da_low_elev.values,
        transform=transform
    ))
)

# Convert low-elevation polygons to GeoDataFrame and remove nulls
gdf_low_elev = gpd.GeoDataFrame.from_features(poly_low_elev, crs=crs_dem)
gdf_low_elev = gdf_low_elev[~gdf_low_elev['val'].isnull()]

# Dissolve all low-elevation polygons into set of disjoint areas
# THINK: doesn't matter what the elevation value is, as long as it's low
gdf_low_elev['val'] = 1
gdf_low_elev = gdf_low_elev.dissolve(by='val').explode('geometry', ignore_index=True)

# Only retain low-elevation polygons intersecting the preliminary AOI
gdf_low_elev = gdf_low_elev[
    gdf_low_elev.to_crs(crs_wgs84).intersects(
        gs_prelim_aoi.union_all()
    )
]

# Clip filtered low-elevation polygons to buffered admin bounds to fill holes
gdf_low_elev = gdf_low_elev.to_crs(crs_wgs84).clip(
    gdf_admin_bounds
        .to_crs(crs_utm50n)
        .buffer(100)
        .to_crs(crs_wgs84)
)

# Dissolve the final result into a single MultiPolygon
gs_low_elev = (
    gdf_low_elev
        .to_crs(crs_utm50n)
        .buffer(50)
        .to_crs(crs_wgs84)
)

# Plot all relevant layers for diagnostics
ax = gs_low_elev.plot()
gdf_admin_bounds.plot(ax=ax, color='lightgrey', alpha=0.4)
gdf_osm_aoi.plot(ax=ax, color='darkgrey', alpha=0.4)
gs_inland.plot(ax=ax, color='red', alpha=0.6)
gs_offshore.plot(ax=ax, color='green', alpha=0.6)
ax.set_xlim(bbox[0] + 0.1, bbox[2] + 0.025)
ax.set_ylim(bbox[1] - 0.001, bbox[3] + 0.025)
ax.set_xticklabels([])
ax.set_yticklabels([])

# Save figure
fp_aoi_portions = plots_dir / f'{municipality[0].replace(" ", "")}_PlotPortionsAOI.png'
plt.tight_layout()
plt.savefig(fp_aoi_portions)


# --- SECTION 4: Combine all portions into final AOI ---


# Combine all into a single MultiPolygon and clip to OSM admin bounds
poly_aoi = gpd.GeoSeries(pd.concat(
    [
        gs_inland, 
        # gs_offshore, 
        gs_low_elev
    ],
    ignore_index=True
)).clip(gdf_osm_aoi.to_crs(crs_wgs84)).to_crs(crs_utm50n).union_all()
# print(poly_aoi)

# Convert to a single Polygon
# if isinstance(poly_aoi, Polygon):
#     poly_aoi = Polygon(poly_aoi.exterior)
# elif isinstance(poly_aoi, MultiPolygon):
#     poly_aoi = MultiPolygon([Polygon(p.exterior.coords) for p in poly_aoi.geoms])
# else:
#     raise ValueError('`poly_aoi` is unexpectedly neither a shapely.Polygon nor shapely.MultiPolygon')

# Delete small holes
def remove_small_holes(geom, area_threshold: int | float):
    """Remove interior holes smaller than the specified area threshold."""
    if isinstance(geom, Polygon):
        # Keep only interiors (holes) whose area is greater than or equal to the threshold
        filtered_interiors = [
            interior for interior in geom.interiors 
            if Polygon(interior).area >= area_threshold
        ]
        return Polygon(geom.exterior, filtered_interiors)
    
    elif isinstance(geom, MultiPolygon):
        # Apply recursively to each polygon in the MultiPolygon
        return MultiPolygon([
            remove_small_holes(poly, area_threshold) 
            for poly in geom.geoms
        ])
    
    return geom

THRESHOLD_SQM = 500_000

poly_aoi = remove_small_holes(poly_aoi, THRESHOLD_SQM)

# Convert to GeoDataFrame for export
gdf_aoi = gpd.GeoDataFrame(
    data=gdf_admin_bounds,
    geometry=gpd.GeoSeries(poly_aoi).make_valid(),
    crs=crs_utm50n
).make_valid().simplify(30, True).to_crs(crs_wgs84)

# Save to GeoJSON
fp_output = output_dir / f'{municipality[0].replace(" ", "")}_AOI_MangroveSeagrassMapping.geojson'
gdf_aoi.to_file(fp_output)