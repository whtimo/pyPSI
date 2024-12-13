import glob

import pandas as pd
import numpy as np
import rasterio
import os
from pathlib import Path
from datetime import datetime
import re


def extract_date_from_filename(filename: str) -> str:
    """
    Extract ISO date from filename

    Parameters:
    -----------
    filename: str
        Filename containing ISO date before .tiff

    Returns:
    --------
    date_str: str
        ISO date string
    """
    # Regular expression to find ISO date pattern before .tiff
    #date_pattern = r'\d{4}-\d{2}-\d{2}(?=\.tiff)'
    date_pattern = r'\d{4}-\d{2}-\d{2}(?=\.topo.interfero.tiff)'
    match = re.search(date_pattern, filename)
    if match:
        return match.group(0)
    raise ValueError(f"No valid date found in filename: {filename}")


def read_complex_phase(file_path: str, pixel_coords: np.ndarray) -> np.ndarray:
    """
    Read phase values from complex interferogram at given coordinates

    Parameters:
    -----------
    file_path: str
        Path to complex interferogram TIFF file
    pixel_coords: np.ndarray
        Array of (sample, line) coordinates

    Returns:
    --------
    phases: np.ndarray
        Phase values at given coordinates
    """
    with rasterio.open(file_path) as src:
        # Read complex values at specified coordinates
        # rasterio expects (row, col) format, so we swap sample/line
        coords = [(line, sample) for sample, line in pixel_coords]
        complex_values = [src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0] for row, col in coords]

        # Convert complex values to phase
        phases = np.angle(complex_values)

    return phases


def extract_ps_phases(ps_csv_path: str,
                      interferogram_dir: str,
                      aps_path: str,
                      output_csv_path: str):
    """
    Extract phase values for PS points, remove atmospheric phase screen (APS),
    and save to CSV

    Parameters:
    -----------
    ps_csv_path: str
        Path to CSV file containing PS coordinates
    interferogram_dir: str
        Directory containing complex interferogram TIFF files
    aps_path: str
        Directory containing APS TIFF files (*_000.tif, *_001.tif, etc.)
    output_csv_path: str
        Path where output CSV will be saved
    """
    # Read PS coordinates
    ps_df = pd.read_csv(ps_csv_path)
    pixel_coords = ps_df[['sample', 'line']].values

    # Get list of interferogram files
    interferogram_files = sorted([
        f for f in os.listdir(interferogram_dir)
        if f.endswith('.tiff')
    ])

    # Initialize dictionary to store results
    # Start with original PS coordinates
    results = {
        'sample': ps_df['sample'],
        'line': ps_df['line']
    }

    # Extract phase values for each interferogram
    for idx, ifg_file in enumerate(interferogram_files):
        ifg_path = Path(interferogram_dir) / ifg_file
        aps_file = Path(aps_path) / f"*_{idx:03d}.tif"

        # Find the matching APS file
        aps_file = glob.glob(str(aps_file))[0]

        # Extract date from filename
        date = extract_date_from_filename(ifg_file)

        # Read phase values from interferogram
        phases = read_complex_phase(str(ifg_path), pixel_coords)

        # Read APS values using interpolation
        with rasterio.open(aps_file) as src:
            # Use bilinear interpolation for floating point coordinates
            aps_values = [
                -float(next(src.sample([(x, y)], 1))[0])
                for x, y in pixel_coords
            ]

        # Convert phases to complex numbers
        complex_phases = np.exp(1j * phases)
        #complex_aps = np.exp(1j * np.array(aps_values))
        complex_aps = np.exp(1j * np.array(aps_values))

        # Subtract APS in complex domain (multiplication by complex conjugate)
        corrected_complex = complex_phases * np.conjugate(complex_aps)

        # Convert back to phase values
        corrected_phases = np.angle(corrected_complex)

        # Add to results dictionary
        results[date] = corrected_phases

    # Create output DataFrame and save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=True)


# Example usage:
if __name__ == "__main__":
    # Define paths
    # PS_CSV_PATH = "path/to/ps_coordinates.csv"
    # INTERFEROGRAM_DIR = "path/to/interferogram/directory"
    # OUTPUT_CSV_PATH = "path/to/output/ps_phases.csv"
    PS_CSV_PATH = "/home/timo/Data/LasVegasDesc2/psc.csv"
    INTERFEROGRAM_DIR = "/home/timo/Data/LasVegasDesc2/topo"
    APS_DIR = "/home/timo/Data/LasVegasDesc2/aps_filtered9"
    OUTPUT_CSV_PATH = "/home/timo/Data/LasVegasDesc2/psc_phases15.csv"

    # Extract phases and save to CSV
    extract_ps_phases(PS_CSV_PATH, INTERFEROGRAM_DIR, APS_DIR, OUTPUT_CSV_PATH)