# Mangrove and seagrass mapping in Hinatuan, Surigao del Sur using PlanetScope imagery

## Service Details

|                       |                               |
|-----------------------|-------------------------------|
|   Service Type        |       Independent Research    |
|   Person in Charge    |   Jerome Matthew L. Maiquez   |
|   Date of Request     |           2026-07-06          |

## Objectives
 
### *Delineate mangrove extent for target years in Surigao del Sur using high-resolution PlanetScope imagery* 

- Develop and assess spectral indices to detect mangrove extent using PlanetScope bands (i.e., VNIR)
- Delineate mangrove extent for target years in Hinatuan, Surigao del Sur using the developed spectral index[^1]
- Apply the mangrove & seagrass detection method to imagery for other years

[^1]: NOTE: either via thresholding or supervised classification, depending on index performance

## Expected Outputs

- Formulas for VNIR spectral indices developed to detect mangrove and seagrass
- Maps of mangrove and seagrass extent for target years in Hinatuan
- Maps of mangrove and seagrass extent change in Hinatuan
- Aggregated statistics per barangay and locally-managed marine protected area (LMMPA)

## In this Repository:
 
- `generate_aoi.py`: Automatically generates the AOI (coastal land and sea) given a city/municipality
- `fabdem.py`: Loads FABDEM data in a STAC-like manner (from `utils.py` of `fabdem-v12` STAC endpoint via Hugging Face (https://huggingface.co/datasets/links-ads/fabdem-v12))
- `plot.py`: Plotting functions for exploratory data analysis of reflectance values and assessment of spectral indices
- `validate.py`: Functions for performing validation outside model accuracy estimation (e.g., IoU, separability metrics, etc.)
- Others to be added in the future...

## Open Questions:

- Perform automated label-free sample annotation? Or rely on external labeled data (e.g., QGIS labeled random sampling model) for now?
- Others to be added in the future...