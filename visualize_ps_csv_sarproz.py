import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize
import pandas as pd
import re
import datetime

def load_points_from_csv(csv_path):
    """
    Load point data from CSV file with reference point in header

    Parameters:
    -----------
    csv_path: str
        Path to the CSV file

    Returns:
    --------
    dict:
        Dictionary containing points data and reference point information
    """


    # Read reference point from first line
    with open(csv_path, 'r') as f:
        ref_line = f.readline()

    # Extract reference point information using regex
    ref_match = re.match(r'Reference Point: id=(\d+) \|\| lat=([\d.-]+) \|\| lon=([\d.-]+)', ref_line)
    ref_id = ref_match.group(1)
    ref_lat = float(ref_match.group(2))
    ref_lon = float(ref_match.group(3))

    # Read the actual data
    df = pd.read_csv(csv_path, skiprows=1)

    # Create results dictionary similar to HDF5 format
    results = {
        'reference_point_id': ref_id,
        'points': {},
        'metadata': {
            'creation_date': str(datetime.datetime.now()),
            'number_of_points': len(df) + 1  # Including reference point
        }
    }

    # Add points from DataFrame
    for _, row in df.iterrows():
        results['points'][str(row['ID'])] = {
            'lat': row['LAT'],
            'lon': row['LON'],
            'height_difference': row['HEIGHT'],
            'velocity_difference': row['Velocity [mm/year]']
        }

    # Add reference point with zero differences
    results['points'][ref_id] = {
        'lat': ref_lat,
        'lon': ref_lon,
        'height_difference': 0.0,
        'velocity_difference': 0.0
    }

    return results


def plot_velocities_on_sar_geo(results,
                               output_path=None,
                               cmap='RdYlBu_r',
                               marker_size=20,
                               dpi=300):
    """
    Plot velocity differences over SAR background image using geographic coordinates

    Parameters:
    -----------
    results: dict
        Results dictionary from load_points_from_csv
    sar_image_path: str
        Path to the geocoded SAR TIFF image
    output_path: str, optional
        Path to save the figure
    cmap: str, optional
        Colormap for velocity values
    marker_size: int, optional
        Size of the scatter plot markers
    dpi: int, optional
        DPI for saved figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from matplotlib.colors import Normalize

    # Read SAR image
    # with rasterio.open(sar_image_path) as src:
    #     sar_image = src.read(1)
    #     transform = src.transform
    #     crs = src.crs

    # Extract coordinates and velocities
    lats = []
    lons = []
    velocities = []

    for point_id, point_data in results['points'].items():
        lats.append(point_data['lat'])
        lons.append(point_data['lon'])
        velocities.append(point_data['velocity_difference'])

    lats = np.array(lats)
    lons = np.array(lons)
    velocities = np.array(velocities)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot SAR background
    # sar_db = 10 * np.log10(np.abs(sar_image) ** 2)
    # sar_norm = Normalize(vmin=np.percentile(sar_db, 5),
    #                      vmax=np.percentile(sar_db, 95))

    # # Plot using geographic extent
    # extent = [
    #     transform.xoff,
    #     transform.xoff + transform.a * sar_image.shape[1],
    #     transform.yoff + transform.e * sar_image.shape[0],
    #     transform.yoff
    # ]

    # plt.imshow(sar_db,
    #            cmap='gray',
    #            norm=sar_norm,
    #            extent=extent)

    # Plot velocities
    scatter = plt.scatter(lons,
                          lats,
                          c=velocities,
                          cmap=cmap,
                          s=marker_size,
                          alpha=0.7)

    # Highlight reference point
    ref_point = results['points'][results['reference_point_id']]
    plt.scatter(ref_point['lon'],
                ref_point['lat'],
                c='yellow',
                marker='*',
                s=marker_size * 2,
                label='Reference Point')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Velocity [mm/year]', rotation=270, labelpad=15)

    # Set labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
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
# Load points from CSV
results = load_points_from_csv('points.csv')

# Create visualization
plot_velocities_on_sar_geo(
    results,
    sar_image_path='sar_image.tiff',
    output_path='velocity_map.png',
    cmap='RdYlBu_r',
    marker_size=20,
    dpi=300
)
"""

results = load_points_from_csv('/home/timo/Data/LasVegasDesc/deftrend.csv')
plot_velocities_on_sar_geo(
    results,
    cmap='RdYlBu_r',
    marker_size=20,
    dpi=300
)
