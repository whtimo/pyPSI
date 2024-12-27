import rasterio
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter


def process_amplitude_dispersion(
        input_tiff_path,
        output_csv_path,
        threshold_value,
        window_size_x,
        window_size_y
):
    """
    Process amplitude dispersion index TIFF file to find local minima within rectangles

    Parameters:
    -----------
    input_tiff_path : str
        Path to the input TIFF file containing amplitude dispersion index
    output_csv_path : str
        Path where the output CSV will be saved
    threshold_value : float
        Threshold value for filtering pixels
    window_size_x : int
        Size of the search window in x direction
    window_size_y : int
        Size of the search window in y direction
    """

    # Read the TIFF file
    with rasterio.open(input_tiff_path) as src:
        # Read the data as a numpy array
        amplitude_data = src.read(1)  # Read first band
        transform = src.transform

        # Apply threshold
        mask = amplitude_data < threshold_value

        # Create a minimum filter with the specified window size
        footprint = np.ones((window_size_y, window_size_x))
        local_min = minimum_filter(amplitude_data, footprint=footprint)

        # Find points that are both below threshold and local minima
        is_local_min = (amplitude_data == local_min) & mask

        # Get coordinates of valid points
        rows, cols = np.where(is_local_min)

        # Convert pixel coordinates to map coordinates
        #xs, ys = rasterio.transform.xy(transform, rows, cols) #Timo: We don't want to transform

        # Create DataFrame with results
        results_df = pd.DataFrame({
            'sample': np.array(cols, dtype=int), # xs,  # Timo changed as the integer coordinate gives the correct reading and avoid interpolation
            'line': np.array(rows, dtype=int), #ys,
            'amplitude_dispersion': amplitude_data[rows, cols]
        })

        # Save to CSV
        results_df.to_csv(output_csv_path, index=True) #added indexing

        return results_df


# Example usage:
if __name__ == "__main__":
    # input_path = "path/to/your/amplitude_dispersion.tif"
    # output_path = "path/to/your/output.csv"
    # threshold = 0.4  # Example threshold value
    # window_x = 5  # Example window size in x direction
    # window_y = 5  # Example window size in y direction

    input_path = "/home/timo/Data/LVS1_snap/amplitude_dispersion.tif"
    output_path = "/home/timo/Data/LVS1_snap/ps.csv"
    threshold = 0.25
    window_x = 3
    window_y = 3

    results = process_amplitude_dispersion(
        input_path,
        output_path,
        threshold,
        window_x,
        window_y
    )