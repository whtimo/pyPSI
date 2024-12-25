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
    sum_amplitude = np.zeros(width, dtype=np.float32)
    sum_squared_amplitude = np.zeros(width, dtype=np.float32)
    amplitude_dispersion = np.zeros(width, dtype=np.float32)

    # Create output file
    with rasterio.open(output_file, 'w', **profile) as dst:
        # Process line by line
        for y in range(height):
            if y % 100 == 0:
                logger.info(f"Processing line {y}/{height}")

            # Reset line sums
            sum_amplitude.fill(0)
            sum_squared_amplitude.fill(0)

            # Process each intensity band for current line
            for band_name in intensity_bands:
                band = product.getBand(band_name)
                line_data = np.zeros(width, dtype=np.float32)
                band.readPixels(0, y, width, 1, line_data)

                # Update running sums
                amplitude = np.sqrt(line_data)  # Convert intensity to amplitude
                sum_amplitude += amplitude
                sum_squared_amplitude += amplitude * amplitude

            # Calculate statistics for current line
            mean_amplitude = sum_amplitude / n_images

            # Calculate variance
            variance = (sum_squared_amplitude / n_images) - (mean_amplitude * mean_amplitude)
            variance = np.maximum(variance, 0)  # Ensure non-negative variance
            std_amplitude = np.sqrt(variance)

            # Calculate amplitude dispersion for current line
            valid_pixels = mean_amplitude > 0
            amplitude_dispersion.fill(np.nan)
            amplitude_dispersion[valid_pixels] = std_amplitude[valid_pixels] / mean_amplitude[valid_pixels]

            # Write line to output file
            dst.write(amplitude_dispersion.reshape(1, -1), 1, window=((y, y + 1), (0, width)))

    # Calculate final statistics
    logger.info("Calculating final statistics...")
    with rasterio.open(output_file) as src:
        amplitude_dispersion_full = src.read(1)
        valid_da = amplitude_dispersion_full[~np.isnan(amplitude_dispersion_full)]
        ps_candidates = np.sum(valid_da < ps_threshold)
        valid_pixels = np.sum(~np.isnan(amplitude_dispersion_full))

    stats = {
        'min_da': float(np.nanmin(valid_da)),
        'max_da': float(np.nanmax(valid_da)),
        'mean_da': float(np.nanmean(valid_da)),
        'median_da': float(np.nanmedian(valid_da)),
        'ps_candidates': int(ps_candidates),
        'total_valid_pixels': int(valid_pixels),
        'processed_images': n_images
    }

    logger.info("Processing completed successfully")
    logger.info(f"Found {ps_candidates} PS candidates (DA < {ps_threshold})")

    # Close the product
    product.dispose()

    return stats


if __name__ == "__main__":
    # Example usage
    input_dim_file = '/home/timo/Data/LVS1_snap/deburst/S1A_IW_SLC__1SDV_20230702T134404_20230702T134432_049245_05EBEA_A4DF_Orb_Stack_esd_deb.dim'
    output_file = '/home/timo/Data/LVS1_snap/amplitude_dispersion.tif'

    try:
        stats = calculate_amplitude_dispersion_snap(input_dim_file, output_file)
        print("Statistics:", stats)
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
