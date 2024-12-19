import numpy as np
import h5py

def save_reference_point(reference_point: str, filename: str = 'reference_point.txt'):
    """
    Save reference point ID to a text file

    Parameters:
    -----------
    reference_point: str
        ID of the reference point
    filename: str
        Path to save the text file
    """
    with open(filename, 'w') as f:
        f.write(reference_point)

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
        'residuals': {},
        'network_edges': {}
    }

    with h5py.File(filename, 'r') as f:
        data_group = f['network_parameters']
        network_group = f['network_edges']

        edge_ids = [id.decode('utf-8') for id in data_group['edge_ids'][:]]
        start_points = [p.decode('utf-8') for p in network_group['start_points'][:]]
        end_points = [p.decode('utf-8') for p in network_group['end_points'][:]]

        for i, edge_id in enumerate(edge_ids):
            # Store parameter data
            params['height_errors'][edge_id] = data_group['height_errors'][i]
            params['velocities'][edge_id] = data_group['velocities'][i]
            params['temporal_coherences'][edge_id] = data_group['temporal_coherences'][i]
            params['residuals'][edge_id] = data_group['residuals'][i]

            # Store network edge information
            params['network_edges'][edge_id] = {
                'start_point': start_points[i],
                'end_point': end_points[i]
            }

    return params


def select_reference_point(params,
                           temporal_coherence_threshold=0.9,
                           min_connections=10,
                           known_stable_points=None):
    """
    Select a reference point based on typical PSInSAR criteria

    Parameters:
    -----------
    params: dict
        Dictionary containing network parameters
    temporal_coherence_threshold: float
        Minimum temporal coherence threshold (default: 0.9)
    min_connections: int
        Minimum number of connecting edges (default: 10)
    known_stable_points: list
        List of point IDs known to be stable (optional)

    Returns:
    --------
    reference_point: str
        ID of selected reference point
    """

    # Count connections for each point
    point_connections = {}
    for edge_id in params['temporal_coherences'].keys():
        start_point = params['network_edges'][edge_id]['start_point']
        end_point = params['network_edges'][edge_id]['end_point']

        # Count connections
        point_connections[start_point] = point_connections.get(start_point, 0) + 1
        point_connections[end_point] = point_connections.get(end_point, 0) + 1

    # Calculate average temporal coherence for each point
    point_coherence = {}
    for edge_id, coherence in params['temporal_coherences'].items():
        start_point = params['network_edges'][edge_id]['start_point']
        end_point = params['network_edges'][edge_id]['end_point']

        if start_point not in point_coherence:
            point_coherence[start_point] = []
        if end_point not in point_coherence:
            point_coherence[end_point] = []

        point_coherence[start_point].append(coherence)
        point_coherence[end_point].append(coherence)

    # Calculate mean coherence for each point
    mean_coherence = {
        point: np.mean(coherences)
        for point, coherences in point_coherence.items()
    }

    # Filter points based on criteria
    candidate_points = []
    for point in mean_coherence.keys():
        # Check if point meets all criteria
        if (mean_coherence[point] >= temporal_coherence_threshold and
                point_connections[point] >= min_connections and
                (known_stable_points is None or point in known_stable_points)):
            candidate_points.append({
                'point_id': point,
                'coherence': mean_coherence[point],
                'connections': point_connections[point]
            })

    # Sort candidates by coherence and connections
    sorted_candidates = sorted(
        candidate_points,
        key=lambda x: (x['coherence'], x['connections']),
        reverse=True
    )

    if not sorted_candidates:
        raise ValueError("No points meet the reference point criteria")

    # Return the best candidate
    return sorted_candidates[0]['point_id']


#params = load_network_parameters('filename')
params = load_network_parameters('/home/timo/Data/LasVegasDesc/psc_results_lambda.h5')
reference_point = select_reference_point(params, temporal_coherence_threshold=0.9, min_connections=10)
save_reference_point(reference_point, '/home/timo/Data/LasVegasDesc/ref_point_lambda.txt')