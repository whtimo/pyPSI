import numpy as np
from scipy import stats
import numpy.ma as ma
from scipy.ndimage import binary_dilation, generate_binary_structure
import os
import rasterio
from pathlib import Path
import multiprocessing as mp
import time

def save_shp_counts(shp_counts, reference_tiff, output_path):
    """
    Save the SHP counts as a GeoTIFF file, preserving geospatial information
    from a reference TIFF.

    Parameters:
    -----------
    shp_counts : numpy.ndarray
        2D array containing the SHP counts for each pixel
    reference_tiff : str or Path
        Path to one of the original TIFF files to copy geospatial information from
    output_path : str or Path
        Path where to save the output GeoTIFF file
    """

    # Convert paths to Path objects
    reference_tiff = Path(reference_tiff)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read metadata from reference file
    with rasterio.open(reference_tiff) as src:
        # Copy the metadata
        metadata = src.meta.copy()

        # Update metadata for the new file
        metadata.update({
            'driver': 'GTiff',
            'dtype': 'uint16',  # Assuming counts won't exceed 65535
            'count': 1,  # Single band
            'nodata': 0  # Set nodata value
        })

        # Write the SHP counts to a new GeoTIFF
        with rasterio.open(output_path, 'w', **metadata) as dst:
            # Convert to uint16 and write
            dst.write(shp_counts.astype('uint16'), 1)

            # Add description
            dst.update_tags(TIFFTAG_IMAGEDESCRIPTION='SHP Counts from DespecKS')

            # Add band description
            dst.set_band_description(1, 'Number of Statistically Homogeneous Pixels')


def save_shp_counts_simple(shp_counts, output_path):
    """
    Simplified version to save the SHP counts as a GeoTIFF file
    without preserving geospatial information.

    Parameters:
    -----------
    shp_counts : numpy.ndarray
        2D array containing the SHP counts for each pixel
    output_path : str or Path
        Path where to save the output GeoTIFF file
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    metadata = {
        'driver': 'GTiff',
        'height': shp_counts.shape[0],
        'width': shp_counts.shape[1],
        'count': 1,
        'dtype': 'uint16',
        'crs': None,
        'transform': rasterio.Affine(1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0),
        'nodata': 0
    }

    # Write the file
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(shp_counts.astype('uint16'), 1)
        dst.update_tags(TIFFTAG_IMAGEDESCRIPTION='SHP Counts from DespecKS')
        dst.set_band_description(1, 'Number of Statistically Homogeneous Pixels')



def load_amplitude_data(input_directory):
    """
    Load amplitude data from complex float32 TIFF files in a directory.

    Parameters:
    -----------
    input_directory : str
        Path to the directory containing the TIFF files

    Returns:
    --------
    amplitude_stack : numpy.ndarray
        3D array of amplitude values (n_images, rows, cols)
    """

    # Convert input directory to Path object
    input_dir = Path(input_directory)

    # Get all tiff files in the directory
    tiff_files = sorted([f for f in input_dir.glob('*.tiff')] +
                        [f for f in input_dir.glob('*.tif')])

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_directory}")

    # Read first image to get dimensions
    with rasterio.open(tiff_files[0]) as src:
        rows = src.height
        cols = src.width

        # Verify that the data type is complex float32
        if src.dtypes[0] != 'complex64':
            raise ValueError("Input TIFF files must be complex float32")

    # Initialize the amplitude stack
    n_images = len(tiff_files)
    amplitude_stack = np.zeros((n_images, rows, cols), dtype=np.float32)

    # Read each image and compute amplitude
    for idx, tiff_file in enumerate(tiff_files):
        with rasterio.open(tiff_file) as src:
            # Read complex data
            complex_data = src.read(1)

            # Compute amplitude (magnitude of complex numbers)
            amplitude_stack[idx] = np.abs(complex_data)

            # Verify image dimensions
            if complex_data.shape != (rows, cols):
                raise ValueError(f"Image {tiff_file.name} has different dimensions")

    return amplitude_stack


def select_shp(amplitude_stack, center_pixel, window_size=20, alpha=0.05, connectivity=8):
    """
    Select Statistically Homogeneous Pixels (SHP) using DespecKS approach,
    ensuring connectivity to the center pixel.

    Parameters:
    -----------
    amplitude_stack : numpy.ndarray
        3D array of amplitude values (n_images, rows, cols)
    center_pixel : tuple
        (row, col) coordinates of the pixel of interest
    window_size : int
        Size of the search window (odd number)
    alpha : float
        Significance level for the KS test
    connectivity : int
        Either 4 or 8, specifying the connectivity type

    Returns:
    --------
    shp_mask : numpy.ndarray
        Boolean mask indicating selected SHPs
    """
    if connectivity not in [4, 8]:
        raise ValueError("Connectivity must be either 4 or 8")

    # Extract window boundaries
    row, col = center_pixel
    half_window = window_size // 2
    row_start = max(0, row - half_window)
    row_end = min(amplitude_stack.shape[1], row + half_window + 1)
    col_start = max(0, col - half_window)
    col_end = min(amplitude_stack.shape[2], col + half_window + 1)

    # Get amplitude time series for center pixel
    center_amplitudes = amplitude_stack[:, row, col]

    # Initialize mask for statistically homogeneous pixels
    stat_homo_mask = np.zeros((row_end - row_start, col_end - col_start), dtype=bool)

    # Initialize relative center position in the window
    center_row_rel = row - row_start
    center_col_rel = col - col_start

    # First pass: identify all statistically homogeneous pixels
    for i in range(row_end - row_start):
        for j in range(col_end - col_start):
            # Get absolute coordinates
            abs_i = i + row_start
            abs_j = j + col_start

            # Skip center pixel
            if (abs_i == row and abs_j == col):
                stat_homo_mask[i, j] = True
                continue

            # Get amplitude time series for current pixel
            test_amplitudes = amplitude_stack[:, abs_i, abs_j]

            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(center_amplitudes, test_amplitudes)

            # If p-value is greater than alpha, pixels are considered homogeneous
            if p_value > alpha:
                stat_homo_mask[i, j] = True

    # Create connectivity structure
    if connectivity == 4:
        struct = generate_binary_structure(2, 1)  # 4-connectivity
    else:
        struct = generate_binary_structure(2, 2)  # 8-connectivity

    # Initialize seed mask with center pixel
    seed_mask = np.zeros_like(stat_homo_mask)
    seed_mask[center_row_rel, center_col_rel] = True

    # Use binary dilation with connectivity constraint
    connected_mask = np.zeros_like(stat_homo_mask)
    temp_mask = seed_mask.copy()

    while True:
        # Dilate the temporary mask
        dilated = binary_dilation(temp_mask, structure=struct)
        # Only keep dilated pixels that are also in stat_homo_mask
        new_mask = dilated & stat_homo_mask
        # If no new pixels were added, break
        if np.array_equal(new_mask, connected_mask):
            break
        connected_mask = new_mask
        temp_mask = new_mask

    return connected_mask

def process_point(point_data):
    i, j = point_data
    if i < 50 and j < 50:
        shp_mask = select_shp(amplitude_stack, (i, j), window_size=20, connectivity=4)
        shp_count = np.sum(shp_mask)
        return i, j, shp_count
    else:
        return i, j, 0

def process_ds_candidates(amplitude_stack, min_shp_count=10):
    """
    Process entire image to identify DS candidates.

    Parameters:
    -----------
    amplitude_stack : numpy.ndarray
        3D array of amplitude values (n_images, rows, cols)
    min_shp_count : int
        Minimum number of SHPs required to consider a pixel as DS candidate

    Returns:
    --------
    ds_candidates : numpy.ndarray
        Boolean mask of DS candidates
    shp_counts : numpy.ndarray
        Number of SHPs for each pixel
    """

    rows, cols = amplitude_stack.shape[1:]
    ds_candidates = np.zeros((rows, cols), dtype=bool)
    shp_counts = np.zeros((rows, cols), dtype=int)

    # Prepare data for parallel processing
    point_data = [(i, j) for i in range(rows) for j in range(cols)]

    # Parallel processing of points
    with mp.Pool() as pool:
        results = pool.map(process_point, point_data)

    for i, j, shp_count in results:
        shp_counts[i, j] = shp_count
        if shp_count >= min_shp_count:
            ds_candidates[i, j] = True

    return ds_candidates, shp_counts


# Example usage:
"""
# Load your amplitude stack (n_images, rows, cols)
amplitude_stack = load_amplitude_data()  # Your data loading function

# Process single pixel
center_pixel = (100, 100)  # Example coordinates
shp_mask = select_shp(amplitude_stack, center_pixel)

# Process entire image
ds_candidates, shp_counts = process_ds_candidates(amplitude_stack)
"""

amplitude_stack = load_amplitude_data('/home/timo/Projects/WuhanTSXAirport/resample')
#center_pixel = (100, 100)  # Example coordinates
#shp_mask = select_shp(amplitude_stack, center_pixel, window_size=20)

start_time = time.perf_counter()
ds_candidates, shp_counts = process_ds_candidates(amplitude_stack)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

save_shp_counts_simple(shp_counts, '/home/timo/Projects/WuhanTSXAirport/shpcount.tiff')
