import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Union
import logging


def calculate_amplitude_dispersion(
        input_directory: str,
        output_file: str,
        ps_threshold: float = 0.25
) -> Dict[str, Union[float, int]]:
    """
    Calculate amplitude dispersion index from a stack of complex SAR images.

    Parameters:
    -----------
    input_directory : str
        Path to directory containing coregistered complex SAR images in GeoTIFF format
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

    # Get list of input files
    image_files = list(Path(input_directory).glob('*.tif'))
    if not image_files:
        raise ValueError(f"No TIFF files found in {input_directory}")

    logger.info(f"Found {len(image_files)} images to process")

    # Read metadata from first image
    with rasterio.open(image_files[0]) as src:
        profile = src.profile
        shape = src.shape

    # Initialize arrays for calculations
    sum_amplitude = np.zeros(shape, dtype=np.float32)
    sum_squared_amplitude = np.zeros(shape, dtype=np.float32)
    n_images = 0

    # Process each image
    logger.info("Processing images...")
    for img_file in image_files:
        with rasterio.open(img_file) as src:
            # Read complex data
            complex_data = src.read(1).astype(np.complex64)

            # Calculate amplitude
            amplitude = np.abs(complex_data)

            # Update running sums
            sum_amplitude += amplitude
            sum_squared_amplitude += amplitude * amplitude
            n_images += 1

        logger.info(f"Processed {n_images}/{len(image_files)} images")

    # Calculate statistics
    mean_amplitude = sum_amplitude / n_images

    # Calculate variance using computational formula
    variance = (sum_squared_amplitude / n_images) - (mean_amplitude * mean_amplitude)
    # Ensure non-negative variance due to numerical precision
    variance = np.maximum(variance, 0)
    std_amplitude = np.sqrt(variance)

    # Calculate amplitude dispersion
    amplitude_dispersion = np.zeros_like(mean_amplitude)
    valid_pixels = mean_amplitude > 0
    amplitude_dispersion[valid_pixels] = std_amplitude[valid_pixels] / mean_amplitude[valid_pixels]

    # Set invalid pixels to NaN
    amplitude_dispersion[~valid_pixels] = np.nan

    # Update profile for output
    profile.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': np.nan
    })

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
    input_dir = "./sar_images"  # Directory containing complex SAR images
    output_file = "./amplitude_dispersion.tif"  # Output file path

    try:
        stats = calculate_amplitude_dispersion(input_dir, output_file)
        print("Statistics:", stats)
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")