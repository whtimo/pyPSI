import sys
sys.path.append('/home/timo/.snap/snap-python')
import esa_snappy
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize
import pandas as pd
import re
import datetime

def find_matching_point_index(reference_point_file, first_csv, second_csv):
    # Read the reference point ID from the text file
    with open(reference_point_file, 'r') as f:
        point_id = int(f.read().strip())

    # Read both CSV files using pandas

    df1 = pd.read_csv(first_csv)
    df2 = pd.read_csv(second_csv)

    # Get the sample and line coordinates for the reference point
    reference_row = df1.iloc[point_id]
    sample = reference_row['sample']
    line = reference_row['line']

    # Find the matching index in the second CSV file
    matching_index = df2[(df2['sample'] == sample) &
                         (df2['line'] == line)].index

    # Return the first matching index if found, otherwise None
    return matching_index[0] if len(matching_index) > 0 else None

def load_csv_data(velocity_csv):
    """
    Load and combine data from three CSV files containing velocities, samples, and lines

    Parameters:
    -----------
    velocity_csv: str
        Path to CSV file containing velocity data

    Returns:
    --------
    dict:
        Dictionary containing combined point data and reference point information
    """

    # Read reference point from first line (using velocity file)


    # Read the data from all files
    df = pd.read_csv(velocity_csv)


    # Merge dataframes on ID

    # Create results dictionary
    results = {
        'points': {},
    }

    # Add points from combined DataFrame
    for _, row in df.iterrows():
        results['points'][str(int(row['id']))] = {
            'sample': row['sample'],
            'line': row['line'],
            'height': row['height_errors'],
            'velocity': row['velocities'],
            'temporal_coherence': row['temporal_coherence']
        }

    # # Find reference point coordinates in the datasets
    # ref_data_sample = df_samples[df_samples['ID'] == int(ref_id)]['Samples [pix]'].iloc[0]
    # ref_data_line = df_lines[df_lines['ID'] == int(ref_id)]['Lines [pix]'].iloc[0]
    #
    # # Add reference point with zero differences
    # results['points'][ref_id] = {
    #     'sample': ref_data_sample,
    #     'line': ref_data_line,
    #     'lat': ref_lat,
    #     'lon': ref_lon,
    #     'height_difference': 0.0,
    #     'velocity_difference': 0.0
    # }

    return results


def plot_velocities_on_sar(results, ref_point,
                           temporal_coherence_threshold,
                           dim_path,
                           output_path=None,
                           cmap='RdYlBu_r',
                           marker_size=20,
                           dpi=300):
    # Read SAR image
    product = esa_snappy.ProductIO.readProduct(input_dim_file)

    # Get dimensions
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()

    # Get intensity bands
    band_names = product.getBandNames()
    intensity_band = [name for name in band_names if name.startswith('Intensity') and '_mst_' in name]

    if not intensity_band:
        raise ValueError("No intensity master band found in the product")

    band = product.getBand(intensity_band[0])
    line_data = np.zeros((height, width), dtype=np.float32)
    band.readPixels(0, 0, width, height, line_data)

    # Calculate amplitude from complex data
    amplitude = np.sqrt(line_data)

    # Extract coordinates and velocities
    samples = []
    lines = []
    velocities = []

    for point_id, point_data in results['points'].items():
        temp_coh = point_data['temporal_coherence']
        if temp_coh >= temporal_coherence_threshold:
            samples.append(point_data['sample'])
            lines.append(point_data['line'])
            velocities.append(point_data['velocity'])

    samples = np.array(samples)
    lines = np.array(lines)
    velocities = np.array(velocities)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot SAR background
    # Convert complex SAR image to intensity in dB
    sar_db = 10 * np.log10(amplitude ** 2)

    # Normalize SAR image for display
    sar_norm = Normalize(vmin=np.percentile(sar_db, 5),
                         vmax=np.percentile(sar_db, 95))

    plt.imshow(sar_db,
               cmap='gray',
               norm=sar_norm,
               extent=[0, amplitude.shape[1], amplitude.shape[0], 0])

    # Plot velocities
    scatter = plt.scatter(samples,
                          lines,
                          c=velocities,
                          cmap=cmap,
                          vmin=-30,
                          vmax=30,
                          s=marker_size,
                          alpha=0.7)

    # Highlight reference point
    ref_point = results['points'][str(ref_point)]
    plt.scatter(ref_point['sample'],
                ref_point['line'],
                c='green',
                marker='*',
                s=marker_size * 2 * 20,
                label='Reference Point')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Velocity [mm/year]', rotation=270, labelpad=15)

    # Set labels and title
    plt.xlabel('Sample [pix]')
    plt.ylabel('Line [pix]')
    plt.title('PS Velocity Map')
    plt.legend()

    # Add grid
    plt.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Example usage:
"""
# Load and combine data
results = load_and_combine_csv_data(
    velocity_csv='velocity.csv',
    samples_csv='samples.csv',
    lines_csv='lines.csv'
)

# Create visualization
plot_velocities_on_sar(
    results,
    sar_image_path='sar_image.tiff',
    output_path='velocity_map.png',
    cmap='RdYlBu_r',
    marker_size=20,
    dpi=300
)
"""
input_dim_file = '/home/timo/Data/LVS1_snap/subset/subset_0_of_S1A_IW_SLC__1SDV_20230702T134404_20230702T134432_049245_05EBEA_A4DF_Orb_Stack_esd_deb.dim'

ref_point = find_matching_point_index('/home/timo/Data/LVS1_snap/ref_point.txt', '/home/timo/Data/LVS1_snap/psc.csv', '/home/timo/Data/LVS1_snap/ps.csv')

results = load_csv_data('/home/timo/Data/LVS1_snap/ps_results.csv')

plot_velocities_on_sar(
    results,
    ref_point,
    0.8,
    input_dim_file,
    output_path='/home/timo/Data/LVS1_snap/ps_velocity_map_LV_snappy.png',
    cmap='RdYlBu_r',
    marker_size=2,
    dpi=600
)
