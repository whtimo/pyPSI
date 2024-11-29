import numpy as np
import networkx as nx


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
    weights = 1 - np.array(temporal_coherence)

    for (point1, point2), weight in zip(edges, weights):
        G.add_edge(point1, point2, weight=weight)

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


def extract_path_parameters(G, paths, heights, velocities):
    """
    Extract height and velocity differences along optimal paths

    Parameters:
    -----------
    G : networkx.Graph
        Weighted graph representing the PS network
    paths : dict
        Dictionary containing optimal paths from each point to reference
    heights : array-like
        Height values for each PS point
    velocities : array-like
        Velocity values for each PS point

    Returns:
    --------
    path_parameters : dict
        Dictionary containing accumulated height and velocity differences
    """
    path_parameters = {}

    for point, path in paths.items():
        if len(path) > 1:  # Skip reference point
            height_diff = 0
            velocity_diff = 0

            # Calculate cumulative differences along the path
            for i in range(len(path) - 1):
                current = path[i]
                next_point = path[i + 1]

                height_diff += heights[next_point] - heights[current]
                velocity_diff += velocities[next_point] - velocities[current]

            path_parameters[point] = {
                'height_difference': height_diff,
                'velocity_difference': velocity_diff,
                'path': path
            }

    return path_parameters


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

