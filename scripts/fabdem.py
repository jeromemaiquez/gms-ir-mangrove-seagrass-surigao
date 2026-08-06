import re
import requests
import pystac
from odc.stac import load
import geopandas as pd
from tqdm import tqdm


def parse_coordinates_from_name(item_name):
    """
    Parse coordinates from FABDEM item names like 'N00W000_FABDEM_V1-2'
    Returns (lat, lon) tuple
    """
    # Pattern: N79W106 -> N79, W106
    pattern = r"([NS])(\d+)([EW])(\d+)"
    match = re.match(pattern, item_name)

    if not match:
        return None, None

    lat_dir, lat_val, lon_dir, lon_val = match.groups()

    # Convert to decimal degrees
    lat = float(lat_val) if lat_dir == "N" else -float(lat_val)
    lon = float(lon_val) if lon_dir == "E" else -float(lon_val)

    return lat, lon


def get_item_bounds_from_name(item_name):
    """
    Get bounding box for a FABDEM tile from its name
    FABDEM tiles are typically 1x1 degree tiles
    """
    lat, lon = parse_coordinates_from_name(item_name)
    if lat is None or lon is None:
        return None

    return {"minx": lon, "miny": lat, "maxx": lon + 1, "maxy": lat + 1}


def intersects_bbox(item_bounds, bbox):
    """
    Check if item bounds intersect with given bbox
    bbox should be (minx, miny, maxx, maxy)
    """
    if item_bounds is None:
        return False

    # Check for intersection
    return not (
        item_bounds["maxx"] < bbox[0]  # item is to the left
        or item_bounds["minx"] > bbox[2]  # item is to the right
        or item_bounds["maxy"] < bbox[1]  # item is below
        or item_bounds["miny"] > bbox[3]
    )  # item is above


def get_catalog_item_names(catalog_url):
    """
    Get list of item names from catalog without loading full items
    """
    try:
        response = requests.get(catalog_url)
        catalog_data = response.json()

        item_names = []
        for link in catalog_data.get("links", []):
            if link.get("rel") == "item":
                href = link.get("href", "")
                item_name = href.split("/")[-2]  # Get the directory name
                item_names.append(item_name)

        return item_names
    except Exception as e:
        print(f"Error fetching catalog: {e}")
        return []


def filter_items_by_bbox(catalog_url, bbox):
    """
    Efficiently filter STAC items by bbox using item names
    Returns list of item names that intersect with bbox
    """
    print("Fetching item names from catalog...")
    item_names = get_catalog_item_names(catalog_url)
    print(f"Found {len(item_names)} items in catalog")

    intersecting_items = []
    for item_name in item_names:
        item_bounds = get_item_bounds_from_name(item_name)
        if intersects_bbox(item_bounds, bbox):
            intersecting_items.append(item_name)

    print(f"Found {len(intersecting_items)} items intersecting with bbox")
    return intersecting_items


def load_filtered_items(catalog_url, bbox):
    """
    Load only the STAC items that intersect with the given bbox
    Much faster than loading all items
    """
    # First, get the intersecting item names
    intersecting_item_names = filter_items_by_bbox(catalog_url, bbox)

    if not intersecting_item_names:
        print("No items found intersecting with bbox")
        return []
    # Load only the intersecting items
    items = []
    print(f"Loading {len(intersecting_item_names)} intersecting items...")

    for item_name in tqdm(intersecting_item_names, desc='Loading intersecting items...'):
        try:
            # Construct the item URL
            item_url = f"https://huggingface.co/datasets/links-ads/fabdem-v12/raw/main/stac_catalog/{item_name}/{item_name}.json"

            # Load the individual item
            item = pystac.Item.from_file(href=item_url)
            items.append(item)

        except Exception as e:
            print(f"Error loading item {item_name}: {e}")
            continue

    print(f"Successfully loaded {len(items)} items")
    return items