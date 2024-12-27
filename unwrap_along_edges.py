import numpy as np
import networkx as nx
import h5py
import pandas as pd
from numpy import ndarray
import datetime

def load_reference_point(filename: str) -> str:
    """
    Load reference point ID from a text file

    Parameters:
    -----------
    filename: str
        Path to the text file containing the reference point ID

    Returns:
    --------
    str:
        ID of the reference point

    Raises:
    -------
    FileNotFoundError:
        If the reference point file does not exist
    """
    try:
        with open(filename, 'r') as f:
            reference_point = f.read().strip()
        return int(reference_point) #Timo: debug because need an integer
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference point file '{filename}' not found. "
                              "Please ensure the reference point has been saved first.")


def convert_network_parameters(params):
    """
    Convert network parameters from HDF5 format to the format required for PS network analysis

    Parameters:
    -----------
    params: dict
        Dictionary containing the network parameters as loaded from HDF5 file
        Contains 'height_errors', 'velocities', 'temporal_coherences', 'residuals', 'network_edges'

    Returns:
    --------
    tuple:
        (edges, temporal_coherence, heights, velocities, ps_points_map)
        - edges: list of tuples [(point1_idx, point2_idx), ...]
        - temporal_coherence: list of coherence values [float, ...]
        - heights: list of height values [float, ...]
        - velocities: list of velocity values [float, ...]
        - ps_points_map: dict mapping PS point IDs to their indices
    """
    # Create a mapping of PS point IDs to indices
    # ps_points = set()
    # for edge_id in params['network_edges']:
    #     ps_points.add(params['network_edges'][edge_id]['start_point'])
    #     ps_points.add(params['network_edges'][edge_id]['end_point'])
    #
    # ps_points_map = {point_id: idx for idx, point_id in enumerate(sorted(ps_points))}

    # Convert edges to index-based format
    edges = []
    temporal_coherence = []
    edge_heights = []
    edge_velocities = []
    edge_residuals = []

    for edge_id in params['network_edges']:
        start_idx = int(params['network_edges'][edge_id]['start_point'])
        end_idx = int(params['network_edges'][edge_id]['end_point'])

        edges.append((start_idx, end_idx))
        temporal_coherence.append(params['temporal_coherences'][edge_id])
        edge_heights.append(params['height_errors'][edge_id])
        edge_velocities.append(params['velocities'][edge_id])
        edge_residuals.append(params['residuals'][edge_id])

    # # Initialize heights and velocities arrays for all points
    # num_points = len(ps_points_map)
    # heights = [0.0] * num_points
    # velocities = [0.0] * num_points
    # residuals = [ndarray] * num_points
    #
    # # Accumulate height and velocity differences along edges
    # # Note: This is a simple accumulation - you might need to adjust based on your specific needs
    # for i, (start_idx, end_idx) in enumerate(edges):
    #     heights[end_idx] = heights[start_idx] + edge_heights[i]
    #     velocities[end_idx] = velocities[start_idx] + edge_velocities[i]
    #     residuals[end_idx] = edge_residuals[i]

    return edges, temporal_coherence, edge_heights, edge_velocities, edge_residuals

def load_network_parameters(filename):  # Added from previously generated code
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


def create_ps_network(ps_points, edges, temporal_coherence):
    """
    Create a weighted network from PS points and edges

    Parameters:
    -----------
    ps_points : array-like
        Array of PS points coordinates [(x1,y1), (x2,y2), ...]
    edges : array-like
        Array of edge connections [(point1_idx, point2_idx), ...]
    temporal_coherence : array-like
        Temporal coherence values for each edge
    edge_ids : array-like
        IDs for each edge

    Returns:
    --------
    G : networkx.Graph
        Weighted graph representing the PS network
    """
    # Create an empty undirected graph
    G = nx.Graph()

    # Add nodes (PS points)
    for i, point in enumerate(ps_points):
        G.add_node(i, pos=point)

    # Add edges with weights based on temporal coherence
    # Convert temporal coherence to weights (higher coherence = lower weight)
    #weights = 1 - np.array(temporal_coherence)
    weights = (1 - np.array(temporal_coherence)) * (1 - np.array(temporal_coherence)) #Timo: Squared seems to be better. Show both versions in paper
    #weights = [None if a_ > (1-0.5)*(1-0.5) else a_ for a_ in weights] # Possible to remove some edges based on a threshold

    for (point1, point2), weight, edge_id in zip(edges, weights, range(len(edges))):
        G.add_edge(point1, point2, weight=weight, edge_id=edge_id, p1=point1, p2=point2)

    return G


def find_optimal_paths_to_reference(G, reference_point):
    """
    Find optimal paths from all points to the reference point

    Parameters:
    -----------
    G : networkx.Graph
        Weighted graph representing the PS network
    reference_point : int
        Index of the reference point

    Returns:
    --------
    paths : dict
        Dictionary containing optimal paths from each point to reference
    distances : dict
        Dictionary containing accumulated weights along optimal paths
    """
    # Calculate shortest paths using Dijkstra's algorithm
    paths = nx.single_source_dijkstra_path(G, reference_point, weight='weight')
    distances = nx.single_source_dijkstra_path_length(G, reference_point, weight='weight')

    return paths, distances


def extract_path_parameters(G, paths, heights, velocities, residuals):
    """
    Extract height and velocity differences along optimal paths

    Parameters:
    -----------
    G : networkx.Graph
        Weighted graph representing the PS network
    paths : dict
        Dictionary containing optimal paths from each point to reference
    heights : dict
        Height values for each edge (edge_id: height_value)
    velocities : dict
        Velocity values for each edge (edge_id: velocity_value)
    residuals : dict
        Residual values for each edge (edge_id: residual_value)

    Returns:
    --------
    path_parameters : dict
        Dictionary containing accumulated height, velocity, and residual differences
    """
    path_parameters = {}

    for point, path in paths.items():
        if len(path) > 1:  # Skip reference point
            height_diff = 0
            velocity_diff = 0
            residual_diff = 0

            # Calculate cumulative differences along the path
            for i in range(len(path) - 1):
                current = path[i]
                next_point = path[i + 1]

                # Get edge data from the graph
                edge_data = G.get_edge_data(current, next_point)
                edge_id = edge_data['edge_id']  # Assuming edge_id is stored in edge attributes
                p1 = edge_data['p1'] #Timo: Added direction in travelling the path
                p2 = edge_data['p2']

                if current == p1:
                    height_diff += heights[edge_id]
                    velocity_diff += velocities[edge_id]
                    residual_diff += residuals[edge_id]
                elif current == p2:
                    height_diff -= heights[edge_id]
                    velocity_diff -= velocities[edge_id]
                    residual_diff -= residuals[edge_id]

            path_parameters[point] = {
                'height_difference': height_diff,
                'velocity_difference': velocity_diff,
                'residual_difference': residual_diff,
                'path': path
            }

    return path_parameters

def save_path_parameters(path_parameters, df, reference_point_id, filename):
    """
    Save path parameters and point coordinates to HDF5 file

    Parameters:
    -----------
    path_parameters: dict
        Dictionary containing path parameters for each point
    df: pandas.DataFrame
        DataFrame containing point coordinates with columns ['point_id', 'sample', 'line']
    reference_point_id: str
        ID of the reference point
    filename: str
        Path to save the HDF5 file
    """

    # Create coordinate lookup dictionary
    coord_lookup = {row['point_id']: (row['sample'], row['line'])
                   for _, row in df[['point_id', 'sample', 'line']].iterrows()}

    # Prepare data arrays
    point_ids = list(path_parameters.keys()) + [reference_point_id]
    n_points = len(point_ids)

    # Initialize arrays
    samples = np.zeros(n_points, dtype=np.int32)
    lines = np.zeros(n_points, dtype=np.int32)
    heights = np.zeros(n_points, dtype=np.float32)
    velocities = np.zeros(n_points, dtype=np.float32)
    residuals = []

    # Fill arrays with path parameters data
    for i, point_id in enumerate(point_ids[:-1]):  # Exclude reference point
        samples[i], lines[i] = coord_lookup[point_id]
        heights[i] = path_parameters[point_id]['height_difference']
        velocities[i] = path_parameters[point_id]['velocity_difference']
        residuals.append(path_parameters[point_id]['residual_difference'])

    # Add reference point (last index)
    ref_idx = n_points - 1
    samples[ref_idx], lines[ref_idx] = coord_lookup[reference_point_id]
    # Reference point parameters are already zero from initialization
    residuals.append(np.zeros(len(residuals[0])))
    # Save to HDF5 file
    with h5py.File(filename, 'w') as f:
        # Create main group
        results = f.create_group('path_parameters')

        # Store point IDs as ASCII strings
        #dt = h5py.special_dtype(vlen=str) #Timo: don't want strings, but int
        point_ids_dataset = results.create_dataset('point_ids', (n_points,), dtype=int)
        point_ids_dataset[:] = point_ids

        # Store coordinates and parameters
        results.create_dataset('sample', data=samples)
        results.create_dataset('line', data=lines)
        results.create_dataset('height_difference', data=heights)
        results.create_dataset('velocity_difference', data=velocities)
        results.create_dataset('residual_difference', data=residuals)

        # Store reference point ID as attribute
        results.attrs['reference_point_id'] = reference_point_id

        # Store metadata
        results.attrs['creation_date'] = str(datetime.datetime.now())
        results.attrs['number_of_points'] = n_points

# Example usage:
"""
# Assuming you have these variables:
ps_points = [(x1,y1), (x2,y2), ...]  # PS point coordinates
edges = [(0,1), (1,2), ...]  # Edge connections
temporal_coherence = [0.8, 0.7, ...]  # Temporal coherence for each edge
heights = [100, 102, ...]  # Height values for each point
velocities = [-2.1, -1.9, ...]  # Velocity values for each point
reference_point = 0  # Index of reference point

# Create the network
G = create_ps_network(ps_points, edges, temporal_coherence)

# Find optimal paths
paths, distances = find_optimal_paths_to_reference(G, reference_point)

# Extract parameters along paths
path_parameters = extract_path_parameters(G, paths, heights, velocities)
"""

print("Load data")
params = load_network_parameters('/home/timo/Data/LVS1_snap/ps_results.h5')
df = pd.read_csv('/home/timo/Data/LVS1_snap/psc.csv')
# Rename the unnamed first column to 'point_id'
df = df.rename(columns={df.columns[0]: 'point_id'})
# Extract coordinates
ps_points = df[['sample', 'line']].values
reference_point = load_reference_point('/home/timo/Data/LVS1_snap/ref_point.txt')

edges, temporal_coherence, heights, velocities, residuals = convert_network_parameters(params)

print("Create Network")
G = create_ps_network(ps_points, edges, temporal_coherence)
print("Find optimal path")
paths, distances = find_optimal_paths_to_reference(G, reference_point)
print("Extract path parameters")
path_parameters = extract_path_parameters(G, paths, heights, velocities, residuals)
print('Save results')
save_path_parameters(path_parameters, df, reference_point, '/home/timo/Data/LVS1_snap/ps_results_path.h5')


