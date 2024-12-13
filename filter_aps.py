import numpy as np
from scipy.ndimage import gaussian_filter
import rasterio
import os
from pathlib import Path


def process_phase_residuals(input_dir, output_dir, spatial_filter_size):
    """
    Process phase residuals with spatial and temporal filtering

    Parameters:
    input_dir: str - Directory containing input TIFF files
    output_dir: str - Directory for output TIFF files
    spatial_filter_size: float - Size of the Gaussian low-pass filter
    """

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all tiff files in input directory
    tiff_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])

    # First pass: Load all images to get dimensions and prepare temporal processing
    first_image = rasterio.open(os.path.join(input_dir, tiff_files[0]))
    height, width = first_image.shape
    n_images = len(tiff_files)

    # Create 3D array to store all images
    phase_stack = np.zeros((n_images, height, width), dtype=np.float32)

    # Load all images
    for i, tiff_file in enumerate(tiff_files):
        with rasterio.open(os.path.join(input_dir, tiff_file)) as src:
            phase_stack[i] = src.read(1)
            profile = src.profile  # Save profile for writing output

    #Timo: Changed as first we have the high-pass filter

    # Temporal filtering: High-pass filter in time domain
    # Calculate temporal mean
    temporal_mean = np.mean(phase_stack, axis=0)

    # Remove temporal mean (high-pass filtering)
    phase_stack = phase_stack - temporal_mean[np.newaxis, :, :]

    # Spatial filtering: Apply Gaussian low-pass filter to each image
    for i in range(n_images):
        print(f'Processing image {i} of {n_images}') #Timo: Add some information for long time processing
        phase_stack[i] = gaussian_filter(phase_stack[i], sigma=spatial_filter_size)



    # Wrap phases between -π and π
    wrapped_phases = np.angle(np.exp(1j * phase_stack))

    # Save processed images
    #profile.update(dtype=rasterio.float32)

    for i, tiff_file in enumerate(tiff_files):
        output_path = os.path.join(output_dir, f"processed_{tiff_file}")
        with rasterio.open(output_path, 'w',
                           driver='GTiff',
                           height=height,
                           width=width,
                           count=1,
                           dtype=np.float32,
        ) as dst:
            dst.write(wrapped_phases[i].astype(np.float32), 1)

# Example usage:
# process_phase_residuals(
#     input_dir='path/to/input/directory',
#     output_dir='path/to/output/directory',
#     spatial_filter_size=5.0
# )

process_phase_residuals(
    input_dir='/home/timo/Data/LasVegasDesc2/aps3',
    output_dir='/home/timo/Data/LasVegasDesc2/aps_filtered8',
    spatial_filter_size=20.0
)