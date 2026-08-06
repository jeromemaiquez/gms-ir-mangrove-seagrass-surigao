# Mangrove and seagrass mapping in Hinatuan, Surigao del Sur using PlanetScope imagery

## Service Details

|                       |                               |
|-----------------------|-------------------------------|
|   Service Type        |       Independent Research    |
|   Person in Charge    |   Jerome Matthew L. Maiquez   |
|   Date of Request     |           2026-07-06          |

## Objectives
 
### *Delineate mangrove and seagrass extent for target years in Surigao del Sur using high-resolution PlanetScope imagery* 

- Develop a spectral index to detect (a) mangrove and (b) seagrass using PlanetScope bands (i.e., VNIR)
- Delineate mangrove and seagrass extent for target years in Hinatuan, Surigao del Sur using the developed spectral index[^1]
- Apply the mangrove & seagrass detection method to imagery for other years

[^1]: NOTE: either via thresholding or supervised classification, depending on index performance

## Expected Outputs

- Formulas for VNIR spectral indices developed to detect mangrove and seagrass
- Maps of mangrove and seagrass extent for target years in Hinatuan
- Maps of mangrove and seagrass extent change in Hinatuan
- Aggregated statistics per barangay and locally-managed marine protected area (LMMPA)

## In this Repository:

- `generate_aoi.py`: Automatically generates the AOI (coastal land and sea) given a city/municipality
- `run_t-test.py`: Runs independent-sample t-tests to measure the differentiability of land cover classes for different spectral indices
- `spectral_signature.py`: Visualizes the spectral signature of different land cover classes (for initial data exploration)
- Others to be added in the future...