import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize


def load_path_parameters(filename='path_parameters.h5'):
    """
    Load path parameters from HDF5 file

    Parameters:
    -----------
    filename: str
        Path to the HDF5 file

    Returns:
    --------
    dict:
        Dictionary containing all path parameters and point information
    """
    import h5py

    results = {}

    with h5py.File(filename, 'r') as f:
        params = f['path_parameters']

        # Get all point IDs
        point_ids = [pid.decode('utf-8') if isinstance(pid, bytes)
                     else pid for pid in params['point_ids'][:]]

        # Get reference point ID
        results['reference_point_id'] = params.attrs['reference_point_id']

        # Create points dictionary
        results['points'] = {}
        for i, point_id in enumerate(point_ids):
            results['points'][point_id] = {
                'sample': params['sample'][i],
                'line': params['line'][i],
                'height_difference': params['height_difference'][i],
                'velocity_difference': params['velocity_difference'][i],
                'residual_difference': params['residual_difference'][i]
            }

        # Store metadata
        results['metadata'] = {
            'creation_date': params.attrs['creation_date'],
            'number_of_points': params.attrs['number_of_points']
        }

    return results

def plot_velocities_on_sar(results, sar_image_path,
                           output_path=None,
                           cmap='RdYlBu_r',
                           marker_size=20,
                           dpi=300):
    """
    Plot velocity differences over SAR background image

    Parameters:
    -----------
    results: dict
        Results dictionary from load_path_parameters
    sar_image_path: str
        Path to the complex SAR TIFF image
    output_path: str, optional
        Path to save the figure. If None, display only
    cmap: str, optional
        Colormap for velocity values
    marker_size: int, optional
        Size of the scatter plot markers
    dpi: int, optional
        DPI for saved figure
    """


    # Read SAR image
    with rasterio.open(sar_image_path) as src:
        sar_image = src.read(1)  # Read first band
        transform = src.transform

    # Extract coordinates and velocities
    samples = []
    lines = []
    velocities = []

    for point_id, point_data in results['points'].items():
        samples.append(point_data['sample'])
        lines.append(point_data['line'])
        velocities.append(point_data['velocity_difference'])

    samples = np.array(samples)
    lines = np.array(lines)
    velocities = np.array(velocities)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot SAR background
    # Convert complex SAR image to intensity in dB
    sar_db = 10 * np.log10(np.abs(sar_image) ** 2)

    # Normalize SAR image for display
    sar_norm = Normalize(vmin=np.percentile(sar_db, 5),
                         vmax=np.percentile(sar_db, 95))

    plt.imshow(sar_db,
               cmap='gray',
               norm=sar_norm,
               extent=[0, sar_image.shape[1], sar_image.shape[0], 0])

    # Plot velocities
    scatter = plt.scatter(samples,
                          lines,
                          c=velocities,
                          cmap=cmap,
                          s=marker_size,
                          alpha=0.7,
                          vmin=-30, vmax=30)

    # Highlight reference point
    ref_point = results['points'][results['reference_point_id']]
    plt.scatter(ref_point['sample'],
                ref_point['line'],
                c='green', #Timo: for better visibility
                marker='*',
                s=marker_size * 20,
                label='Reference Point')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Velocity Difference [mm/year]', rotation=270, labelpad=15)

    # Set labels and title
    plt.xlabel('Sample')
    plt.ylabel('Line')
    plt.title('PS Velocity Differences')
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
# Load results
results = load_path_parameters('path_parameters.h5')

# Plot
plot_velocities_on_sar(
    results,
    sar_image_path='sar_image.tiff',
    output_path='velocity_map.png',
    cmap='RdYlBu_r',  # Red-Yellow-Blue colormap, reversed
    marker_size=20,
    dpi=300
)
"""

results = load_path_parameters('/home/timo/Data/LasVegasDesc/ps_results_lambda_path.h5')
plot_velocities_on_sar(
    results,
    sar_image_path='/home/timo/Data/LasVegasDesc/resampled/TSX-1_0_2010-09-19.tiff',
    output_path='/home/timo/Data/LasVegasDesc/psc_parameters_lambda.png',
    marker_size=5, dpi=1200)