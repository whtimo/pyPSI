import esa_snappy
import pandas as pd
import numpy as np
from datetime import datetime

def extract_date_from_band_name(bandname: str) -> str:

    # Regular expression to find ISO date pattern before .tiff
    #date_pattern = r'\d{4}-\d{2}-\d{2}(?=\.tiff)'
    date_string = bandname[-9:]
    parsed_date = datetime.strptime(date_string, "%d%b%Y")
    iso_date = parsed_date.strftime("%Y-%m-%d")

    return iso_date

def read_complex_phase_snappy(product, band_name: str, width : int, height: int, pixel_coords: np.ndarray) -> np.ndarray:

    band = product.getBand(band_name)
    data = np.zeros((height, width), dtype=np.float32)
    band.readPixels(0, 0, width, height, data)

    # Read complex values at specified coordinates
    # rasterio expects (row, col) format, so we swap sample/line
    coords = [(line, sample) for sample, line in pixel_coords]
    phase_values = []
    for coord in coords:
        phase_values.append(data[coord[0], coord[1]])

    # Convert complex values to phase
    phases = np.array(phase_values)

    return phases


def extract_ps_phases(ps_csv_path: str,
                      input_dim_file: str,
                      output_csv_path: str):

    # Read PS coordinates
    ps_df = pd.read_csv(ps_csv_path)
    pixel_coords = ps_df[['sample', 'line']].values

    # Initialize dictionary to store results
    # Start with original PS coordinates
    results = {
        'sample': ps_df['sample'],
        'line': ps_df['line']
    }

    product = esa_snappy.ProductIO.readProduct(input_dim_file)

    # Get dimensions
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()

    # Get intensity bands
    band_names = product.getBandNames()
    phase_bands = [name for name in band_names if name.startswith('Phase')]

    if not phase_bands:
        raise ValueError("No intensity bands found in the product")

    n_images = len(phase_bands)

    dates_bands = []
    # Extract phase values for each interferogram
    for phase_band in phase_bands:

        # Extract date from filename
        date = extract_date_from_band_name(phase_band)
        dates_bands.append((date, phase_band))

    sorted_dates_bands = sorted(dates_bands, key=lambda date_band: date_band[0])

    for date, phase_band in sorted_dates_bands:
        # Read phase values
        phases = read_complex_phase_snappy(product, phase_band, width, height, pixel_coords)

        # Add to results dictionary
        results[date] = phases

    # Create output DataFrame and save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=True) # manually changed index to True


# Example usage:
if __name__ == "__main__":
    # Define paths

    OUTPUT_CSV_PATH = ""
    PS_CSV_PATH = "./psc.csv" #Timo: CSV file with the selected PSC points
    input_dim_file = './subset_0_of_S1A_IW_SLC__1SDV_20230702T134404_20230702T134432_049245_05EBEA_A4DF_Orb_Stack_esd_deb_ifg.dim'
    # Path to BEAM-DIMAP stack including the topo-removed interferograms
    OUTPUT_CSV_PATH = "./psc_phases.csv" #Timo: Output

    # Extract phases and save to CSV
    extract_ps_phases(PS_CSV_PATH, input_dim_file, OUTPUT_CSV_PATH)