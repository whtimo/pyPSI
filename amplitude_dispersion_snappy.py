import sys
sys.path.append('/home/timo/.snap/snap-python')
import numpy as np
from pathlib import Path
from typing import Dict, Union
import logging
import esa_snappy
import rasterio


def calculate_amplitude_dispersion_snap(
        input_dim_file: str,
        output_file: str,
        ps_threshold: float = 0.25
) -> Dict[str, Union[float, int]]:
    """
    Calculate amplitude dispersion index from a BEAM-DIMAP file containing intensity bands.

    Parameters:
    -----------
    input_dim_file : str
        Path to the input BEAM-DIMAP (.dim) file
    output_file : str
        Path where the output amplitude dispersion GeoTIFF will be saved
    ps_threshold : float
        Threshold for PS candidate selection (default: 0.25)

    Returns:
    --------
    Dict containing statistics about the amplitude dispersion calculation
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Read the product
    logger.info(f"Reading product: {input_dim_file}")
    product = esa_snappy.ProductIO.readProduct(input_dim_file)

    # Get dimensions
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()

    # Get intensity bands
    band_names = product.getBandNames()
    intensity_bands = [name for name in band_names if name.startswith('Intensity')]

    if not intensity_bands:
        raise ValueError("No intensity bands found in the product")

    n_images = len(intensity_bands)
    logger.info(f"Found {n_images} intensity bands to process")

    # Get geotransform and projection information from the first band
    first_band = product.getBand(intensity_bands[0])

    # Create output profile for rasterio
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'float32',
        'nodata': np.nan #,
        #'crs': product.getSceneCRS().toString(),  # Convert CRS to string
        #'transform': product.getSceneGeoCoding().getImageToMapTransform()  # Get transformation
    }

    # Initialize arrays for line-by-line processing
    sum_amplitude = np.zeros((height, width), dtype=np.float32)
    sum_squared_amplitude = np.zeros((height, width), dtype=np.float32)
    amplitude_dispersion = np.zeros((height, width), dtype=np.float32)

    for band_name in intensity_bands:
        print("processing band", band_name)
        band = product.getBand(band_name)
        line_data = np.zeros((height, width), dtype=np.float32)
        band.readPixels(0, 0, width, height, line_data)
        amplitude = np.sqrt(line_data)

        # Update running sums
        sum_amplitude += amplitude
        sum_squared_amplitude += amplitude * amplitude

    # Calculate statistics
    mean_amplitude = sum_amplitude / n_images

    # Calculate variance
    variance = (sum_squared_amplitude / n_images) - (mean_amplitude * mean_amplitude)
    variance = np.maximum(variance, 0)  # Ensure non-negative variance
    std_amplitude = np.sqrt(variance)

    # Calculate amplitude dispersion for current line
    valid_pixels = mean_amplitude > 0
    amplitude_dispersion.fill(np.nan)
    amplitude_dispersion[valid_pixels] = std_amplitude[valid_pixels] / mean_amplitude[valid_pixels]

    # Write output
    logger.info("Writing output file...")
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(amplitude_dispersion.astype(np.float32), 1)

    # Calculate statistics
    valid_da = amplitude_dispersion[~np.isnan(amplitude_dispersion)]
    ps_candidates = np.sum(valid_da < ps_threshold)

    stats = {
        'min_da': float(np.nanmin(amplitude_dispersion)),
        'max_da': float(np.nanmax(amplitude_dispersion)),
        'mean_da': float(np.nanmean(amplitude_dispersion)),
        'median_da': float(np.nanmedian(amplitude_dispersion)),
        'ps_candidates': int(ps_candidates),
        'total_valid_pixels': int(np.sum(valid_pixels)),
        'processed_images': n_images
    }

    logger.info("Processing completed successfully")
    logger.info(f"Found {ps_candidates} PS candidates (DA < {ps_threshold})")

    return stats


if __name__ == "__main__":
    # Example usage
    input_dim_file = '/home/timo/Data/LVS1_snap/subset/subset_0_of_S1A_IW_SLC__1SDV_20230702T134404_20230702T134432_049245_05EBEA_A4DF_Orb_Stack_esd_deb.dim'
    output_file = '/home/timo/Data/LVS1_snap/amplitude_dispersion.tif'

    try:
        stats = calculate_amplitude_dispersion_snap(input_dim_file, output_file)
        print("Statistics:", stats)
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
