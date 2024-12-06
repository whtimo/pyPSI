import numpy as np
import rasterio
from scipy.interpolate import griddata, Rbf, NearestNDInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import h5py
from pathlib import Path

def prepare_point_data(points_dict):
    """
    Extract coordinates and residuals from points dictionary

    Parameters:
    -----------
    points_dict: dict
        Dictionary containing point information

    Returns:
    --------
    samples: np.ndarray
        Array of sample coordinates
    lines: np.ndarray
        Array of line coordinates
    residuals: np.ndarray
        2D array of residual differences, shape (n_points, n_residuals)
    """
    samples = []
    lines = []
    residuals = []

    for point_info in points_dict.values():
        samples.append(point_info['sample'])
        lines.append(point_info['line'])
        residuals.append(point_info['residual_difference'])

    return (np.array(samples),
            np.array(lines),
            np.array(residuals))

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

def interpolate_phase_residuals_griddata(samples, lines, values, grid_size):
    """
    Interpolate phase residuals using different methods available in griddata

    Parameters:
    -----------
    samples : array-like
        Sample coordinates of the PS points
    lines : array-like
        Line coordinates of the PS points
    values : array-like
        Unwrapped phase residual values
    grid_size : tuple
        Size of the output grid (n_lines, n_samples)

    Returns:
    --------
    dict : Interpolated grids using different methods
    """

    # Create regular grid to interpolate the data
    grid_lines, grid_samples = np.mgrid[0:grid_size[0], 0:grid_size[1]]

    # Prepare the known points
    points = np.column_stack((lines, samples))

    # Dictionary to store results
    interpolated = {}

    # Perform interpolation using different methods
    for method in ['linear', 'cubic', 'nearest']:
        interpolated[method] = griddata(points, values,
                                        (grid_lines, grid_samples),
                                        method=method)

    return interpolated


def interpolate_phase_residuals_rbf(samples, lines, values, grid_size):
    """
    Interpolate phase residuals using Radial Basis Functions

    Parameters:
    -----------
    samples : array-like
        Sample coordinates of the PS points
    lines : array-like
        Line coordinates of the PS points
    values : array-like
        Unwrapped phase residual values
    grid_size : tuple
        Size of the output grid (n_lines, n_samples)

    Returns:
    --------
    numpy.ndarray : Interpolated grid
    """

    # Create RBF interpolator
    # You can try different functions: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic'
    rbf = Rbf(samples, lines, values, function='thin_plate')

    # Create regular grid
    grid_lines, grid_samples = np.mgrid[0:grid_size[0], 0:grid_size[1]]

    # Interpolate
    interpolated = rbf(grid_samples, grid_lines)

    return interpolated


def interpolate_phase_residuals_kriging(samples, lines, values, grid_size):
    """
    Interpolate phase residuals using Kriging (Gaussian Process Regression)

    Parameters:
    -----------
    samples : array-like
        Sample coordinates of the PS points
    lines : array-like
        Line coordinates of the PS points
    values : array-like
        Unwrapped phase residual values
    grid_size : tuple
        Size of the output grid (n_lines, n_samples)

    Returns:
    --------
    numpy.ndarray : Interpolated grid
    """

    # Prepare the known points
    X = np.column_stack((samples, lines))

    # Define the kernel (you can adjust the parameters)
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    # Create and fit the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, values)

    # Create grid for prediction
    grid_lines, grid_samples = np.mgrid[0:grid_size[0], 0:grid_size[1]]
    grid_points = np.column_stack((grid_samples.ravel(), grid_lines.ravel()))

    # Predict
    interpolated = gp.predict(grid_points).reshape(grid_size)

    return interpolated


def interpolate_phase_residuals_natural_neighbor(samples, lines, values, grid_size):
    """
    Interpolate phase residuals using Natural Neighbor interpolation

    Parameters:
    -----------
    samples : array-like
        Sample coordinates of the PS points
    lines : array-like
        Line coordinates of the PS points
    values : array-like
        Unwrapped phase residual values
    grid_size : tuple
        Size of the output grid (n_lines, n_samples)

    Returns:
    --------
    numpy.ndarray : Interpolated grid
    """

    # Create interpolator
    interpolator = NearestNDInterpolator(list(zip(samples, lines)), values)

    # Create regular grid
    grid_lines, grid_samples = np.mgrid[0:grid_size[0], 0:grid_size[1]]

    # Interpolate
    interpolated = interpolator(grid_samples, grid_lines)

    return interpolated

def save_interpolated_grid(interpolated_data: np.ndarray,
                           output_path: Path,
                           epoch: int):

    width, height = interpolated_data.shape

    with rasterio.open(
            output_path / f'residual_epoch_{epoch:03d}.tif',
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=None,  # Set your CRS if applicable
            #transform=transform,
    ) as dst:
        dst.write(interpolated_data.astype(np.float32), 1)

master_image_width = 10944  # Timo: add fixed size not based on min/max samples from the PSCs
master_image_height = 6016
grid_size = (master_image_width, master_image_height)

input_file = Path('/home/timo/Data/LasVegasDesc/ps_results3_psc_filt_year_results.h5')
output_dir = Path('/home/timo/Data/LasVegasDesc/aps')
output_dir.mkdir(exist_ok=True)

# Load data
data = load_path_parameters(input_file)

# Prepare point data
samples, lines, residuals = prepare_point_data(data['points'])
n_epochs = residuals.shape[1]
print(f"Processing {n_epochs} epochs")

for epoch in range(n_epochs):
    print(f"Processing epoch {epoch + 1}/{n_epochs}")

    # Get residuals for current epoch
    epoch_residuals = residuals[:, epoch]
    interpolated_nn = interpolate_phase_residuals_natural_neighbor(lines, samples, epoch_residuals, grid_size)
    save_interpolated_grid(
            interpolated_nn, output_dir, epoch
        )

print('end')