import matplotlib.pyplot as plt
import numpy as np
import h5py

def load_network_parameters(filename):
    """
    Load network parameters from HDF5 file

    Parameters:
    -----------
    filename: str
        Path to the HDF5 file

    Returns:
    --------
    params: dict
        Dictionary containing the network parameters
    """
    params = {
        'height_errors': {},
        'velocities': {},
        'temporal_coherences': {},
        'residuals': {}
    }

    with h5py.File(filename, 'r') as f:
        data_group = f['network_parameters']
        edge_ids = [id.decode('utf-8') for id in data_group['edge_ids'][:]]

        for i, edge_id in enumerate(edge_ids):
            params['height_errors'][edge_id] = data_group['height_errors'][i]
            params['velocities'][edge_id] = data_group['velocities'][i]
            params['temporal_coherences'][edge_id] = data_group['temporal_coherences'][i]
            params['residuals'][edge_id] = data_group['residuals'][i]

    return params

def plot_temporal_coherence_histogram(params):
    # Extract temporal coherence values from dictionary
    coherence_values = list(params['temporal_coherences'].values())

    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    # Create histogram
    plt.hist(coherence_values, bins=50, color='blue', alpha=0.7, edgecolor='black')

    # Customize the plot
    plt.title('Histogram of Temporal Coherence Values')
    plt.xlabel('Temporal Coherence')
    plt.ylabel('Frequency')

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add mean and median lines
    mean_coherence = np.mean(coherence_values)
    median_coherence = np.median(coherence_values)

    plt.axvline(mean_coherence, color='red', linestyle='--', label=f'Mean: {mean_coherence:.3f}')
    plt.axvline(median_coherence, color='green', linestyle='--', label=f'Median: {median_coherence:.3f}')

    # Add legend
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


# Load and plot
filename = '/home/timo/Data/LasVegasDesc/ps_results2.h5'
params = load_network_parameters(filename)
plot_temporal_coherence_histogram(params)