import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import Rbf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

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



# Example usage
samples = np.array([...])  # your sample coordinates
lines = np.array([...])  # your line coordinates
values = np.array([...])  # your phase residual values
grid_size = (1000, 1000)  # desired output grid size

# Using griddata
interpolated_grids = interpolate_phase_residuals_griddata(samples, lines, values, grid_size)

# Using RBF
interpolated_rbf = interpolate_phase_residuals_rbf(samples, lines, values, grid_size)

# Using Kriging
interpolated_kriging = interpolate_phase_residuals_kriging(samples, lines, values, grid_size)

# Using Natural Neighbor
interpolated_nn = interpolate_phase_residuals_natural_neighbor(samples, lines, values, grid_size)
