import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, List, Dict, Union
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py


def save_network_parameters(params, ps_network, filename):
    """
    Save network parameters to HDF5 file

    Parameters:
    -----------
    params: dict
        Dictionary containing the network parameters
    ps_network: PSNetwork
        The PS network object containing edge information
    filename: str
        Path to save the HDF5 file
    """
    # Convert dictionary data to arrays
    edge_ids = sorted(list(params['height_errors'].keys()))
    n_edges = len(edge_ids)

    # Create numpy arrays
    height_errors_array = np.array([params['height_errors'][edge_id] for edge_id in edge_ids])
    velocities_array = np.array([params['velocities'][edge_id] for edge_id in edge_ids])
    temporal_coherences_array = np.array([params['temporal_coherences'][edge_id] for edge_id in edge_ids])

    # For residuals, we need to handle the array of arrays
    residual_length = len(next(iter(params['residuals'].values())))
    residuals_array = np.zeros((n_edges, residual_length))
    for i, edge_id in enumerate(edge_ids):
        residuals_array[i, :] = params['residuals'][edge_id]

    # Create arrays for network edge information
    start_points = np.array([ps_network['edges'][edge_id]['start_point'] for edge_id in edge_ids], dtype='S20')
    end_points = np.array([ps_network['edges'][edge_id]['end_point'] for edge_id in edge_ids], dtype='S20')

    # Save edge IDs as strings
    edge_ids_array = np.array(edge_ids, dtype='S10')

    with h5py.File(filename, 'w') as f:
        # Create a group for the data
        data_group = f.create_group('network_parameters')

        # Save the arrays
        data_group.create_dataset('edge_ids', data=edge_ids_array)
        data_group.create_dataset('height_errors', data=height_errors_array)
        data_group.create_dataset('velocities', data=velocities_array)
        data_group.create_dataset('temporal_coherences', data=temporal_coherences_array)
        data_group.create_dataset('residuals', data=residuals_array)

        # Save network edge information
        network_group = f.create_group('network_edges')
        network_group.create_dataset('start_points', data=start_points)
        network_group.create_dataset('end_points', data=end_points)

class PSIParameterEstimator:
    def __init__(self,
                 wavelength: float,
                 temporal_baselines: np.ndarray,
                 perpendicular_baselines: np.ndarray,
                 range_distances: np.ndarray,
                 incidence_angles: np.ndarray):
        """
        Initialize PSI parameter estimator

        Parameters:
        -----------
        wavelength: float
            Radar wavelength in meters
        temporal_baselines: np.ndarray
            Time differences between master and slave images in days
        perpendicular_baselines: np.ndarray
            Perpendicular baselines in meters
        range_distances: np.ndarray
            Slant range distances in meters
        incidence_angles: np.ndarray
            Local incidence angles in radians
        """
        self.wavelength = wavelength
        self.temporal_baselines_years = temporal_baselines / 365.0 #Timo: temporal baselines need to be given in years
        self.perpendicular_baselines = perpendicular_baselines
        self.range_distances = range_distances
        self.incidence_angles = incidence_angles

    def estimate_parameters_along_edge(self, phase_differences):
        """
        Estimates height error and velocity along network edges using periodogram approach.

        Args:
            phase_differences (ndarray): Complex phase differences between two PS points

        Returns:
            tuple: (height_error, velocity, temporal_coherence, residuals)
        """
        # Define search spaces for height error and velocity
        height_search = np.linspace(-100, 100, 200)  # meters, adjust range as needed
        velocity_search = np.linspace(-100, 100, 200)  # mm/year, adjust range as needed

        # Initialize coherence matrix
        coherence_matrix = np.zeros((len(height_search), len(velocity_search)))

        # Calculate constants for phase conversion
        height_to_phase = (4 * np.pi / self.wavelength) * (
                self.perpendicular_baselines / (self.range_distances * np.sin(self.incidence_angles))
        )
        velocity_to_phase = (4 * np.pi / self.wavelength) * self.temporal_baselines_years

        # Compute periodogram
        for i, h in enumerate(height_search):
            for j, v in enumerate(velocity_search):
                # Calculate model phases
                #phase_topo = h * height_to_phase
                #phase_motion = v * velocity_to_phase
                #model_phase = phase_topo + phase_motion
                phase_topo = np.angle(np.exp(1j * h * height_to_phase))
                phase_motion = np.angle(np.exp(1j * (v / 1000.0) * velocity_to_phase)) #Timo: velocity is given in mm
                model_phase = np.angle(np.exp(1j * (phase_topo + phase_motion)))

                # Calculate temporal coherence (equation 6.10)
                # temporal_coherence = np.abs(
                #     np.mean(
                #         np.exp(1j * np.angle(phase_differences)) *
                #         np.exp(-1j * model_phase)
                #     )
                # )

                #Timo: The np.angle of the phase difference seems to be a mistake as these are already given in radians
                temporal_coherence = np.abs(
                    np.mean(
                        np.exp(1j * phase_differences) *
                        np.exp(-1j * model_phase)
                    )
                )
                coherence_matrix[i, j] = temporal_coherence

        # Find maximum coherence
        max_idx = np.unravel_index(np.argmax(coherence_matrix), coherence_matrix.shape)
        best_height = height_search[max_idx[0]]
        best_velocity = velocity_search[max_idx[1]]
        max_coherence = coherence_matrix[max_idx]

        # Calculate residuals
        best_phase_topo = np.angle(np.exp(1j * best_height * height_to_phase))
        best_phase_motion = np.angle(np.exp(1j * (best_velocity / 1000) * velocity_to_phase))
        best_model_phase = np.angle(np.exp(1j * (best_phase_topo + best_phase_motion)))
        temporal_coherence2 = np.abs(
            np.mean(
                np.exp(1j * phase_differences) *
                np.exp(-1j * best_model_phase)
            )
        )
        # residuals = np.angle(
        #     np.exp(1j * np.angle(phase_differences)) *
        #     np.exp(-1j * model_phase)
        # )
        # Timo: The np.angle of the phase difference seems to be a mistake as these are already given in radians
        residuals = np.angle(
            np.exp(1j * phase_differences) *
            np.exp(-1j * best_model_phase)
        )

        return best_height, best_velocity, max_coherence, residuals


class NetworkParameterEstimator:
    def __init__(self, ps_network):
        """
        Estimate parameters for entire PS network

        Parameters:
        -----------
        ps_network: dict
            Network information including edges and phase data
        """
        self.network = ps_network
        self.parameter_estimator = PSIParameterEstimator(
            wavelength=ps_network['wavelength'],
            temporal_baselines=ps_network['temporal_baselines'],
            perpendicular_baselines=ps_network['perpendicular_baselines'],
            range_distances=ps_network['range_distances'],
            incidence_angles=ps_network['incidence_angles']
        )

    def estimate_network_parameters(self) -> dict:
        """
        Estimate parameters for all edges in the network

        Returns:
        --------
        network_parameters: dict
            Dictionary containing estimated parameters for each edge
        """
        network_parameters = {
            'height_errors': {},
            'velocities': {},
            'temporal_coherences': {},  # New dictionary for temporal coherences
            'residuals': {}
        }

        edges = self.network['edges'].items()
        for edge_id, edge_data in edges:

            height_error, velocity, temporal_coherence, residuals = (
                self.parameter_estimator.estimate_parameters_along_edge(
                    edge_data['phase_differences']
                )
            )

            print(f'{edge_id} / {len(edges)} - {height_error},{velocity},{temporal_coherence}')

            network_parameters['height_errors'][edge_id] = height_error
            network_parameters['velocities'][edge_id] = velocity
            network_parameters['temporal_coherences'][edge_id] = temporal_coherence
            network_parameters['residuals'][edge_id] = residuals

        return network_parameters


class PSNetwork:
    def __init__(self, dates: List[datetime], xml_path: str, triangle_file: str, points_file: str):
        self.dates = dates
        self.xml_path = Path(xml_path)
        self._wavelength = None
        self._temporal_baselines = None
        self._perpendicular_baselines = None
        self._range_distances = None
        self._incidence_angles = None
        self._edges = None

        # Store file paths
        self.triangle_file = triangle_file
        self.points_file = points_file

        # Process XML files
        self._process_xml_files()
        # Process network edges
        self._process_network_edges()

    def _process_network_edges(self):
        """Process triangle and points files to create edge data using complex numbers"""
        # Read the triangles and points
        triangles_df = pd.read_csv(self.triangle_file)
        points_df = pd.read_csv(self.points_file, index_col=0)

        # Initialize edges dictionary
        self._edges = {}
        edge_counter = 0

        # Process each triangle
        for _, triangle in triangles_df.iterrows():
            # Get points of the triangle
            p1, p2, p3 = triangle[['point1_id', 'point2_id', 'point3_id']]

            # Get phase data for each point and convert to complex
            date_cols = [d.strftime('%Y-%m-%d') for d in self.dates]
            p1_complex = np.exp(1j * points_df.loc[p1, date_cols].values)
            p2_complex = np.exp(1j * points_df.loc[p2, date_cols].values)
            p3_complex = np.exp(1j * points_df.loc[p3, date_cols].values)

            # Create edges (3 edges per triangle)
            # Edge 1: p1 -> p2
            self._edges[edge_counter] = {
                'start_point': p1,
                'end_point': p2,
                'phase_differences': np.angle(p1_complex * np.conjugate(p2_complex))
            }
            edge_counter += 1

            # Edge 2: p2 -> p3
            self._edges[edge_counter] = {
                'start_point': p2,
                'end_point': p3,
                'phase_differences': np.angle(p2_complex * np.conjugate(p3_complex))
            }
            edge_counter += 1

            # Edge 3: p3 -> p1
            self._edges[edge_counter] = {
                'start_point': p3,
                'end_point': p1,
                'phase_differences': np.angle(p3_complex * np.conjugate(p1_complex))
            }
            edge_counter += 1

    def _process_xml_files(self):
        # Speed of light in meters per second
        SPEED_OF_LIGHT = 299792458.0

        # Initialize arrays with the same length as dates
        n_dates = len(self.dates)
        self._temporal_baselines = np.zeros(n_dates)
        self._perpendicular_baselines = np.zeros(n_dates)
        self._range_distances = np.zeros(n_dates)  # You'll need to add this from XML if available
        self._incidence_angles = np.zeros(n_dates)  # You'll need to add this from XML if available

        # Process each XML file
        for idx, date in enumerate(self.dates):
            # Find corresponding XML file
            xml_file = list(self.xml_path.glob(f"*_{date.strftime('%Y-%m-%d')}.topo.interfero.xml"))
            if not xml_file:
                continue

            # Parse XML
            tree = ET.parse(xml_file[0])
            root = tree.getroot()

            # Extract interferogram attributes
            interferogram = root.find('Interferogram')
            if interferogram is not None:
                self._temporal_baselines[idx] = float(interferogram.get('temp_baseline'))
                self._perpendicular_baselines[idx] = float(interferogram.get('baseline'))

            # Extract wavelength (only needs to be done once)
            if self._wavelength is None:
                wavelength_elem = root.find('.//Wavelength')
                if wavelength_elem is not None:
                    self._wavelength = float(wavelength_elem.text)

            # Find all Grid elements
            grid_elements = root.findall(
                './/Grid')  # Comment timo: I don't like this and think it should only be derive grids from the master not all as this possibly does
            if grid_elements:
                # Find the center grid
                n_grids = len(grid_elements)
                center_grid = grid_elements[n_grids // 2]

                # Extract incidence angle
                incidence_angle_elem = center_grid.find('IncidenceAngle')
                if incidence_angle_elem is not None:
                    self._incidence_angles[idx] = float(incidence_angle_elem.text)

                # Extract range time and convert to distance
                range_time_elem = center_grid.find('RangeTime')
                if range_time_elem is not None:
                    range_time = float(range_time_elem.text)
                    # Convert two-way travel time to one-way distance
                    self._range_distances[idx] = (range_time * SPEED_OF_LIGHT) / 2

    def __getitem__(self, key: str) -> Union[np.ndarray, dict]:
        """Allow dictionary-like access to the network parameters"""
        if key == 'wavelength':
            return self._wavelength
        elif key == 'temporal_baselines':
            return self._temporal_baselines
        elif key == 'perpendicular_baselines':
            return self._perpendicular_baselines
        elif key == 'range_distances':
            return self._range_distances
        elif key == 'incidence_angles':
            return self._incidence_angles
        elif key == 'edges':
            return self._edges
        else:
            raise KeyError(f"Key {key} not found in PSNetwork")

# Read the CSV file
#df = pd.read_csv('your_file.csv')
df = pd.read_csv('./psc_phases.csv')

# Get the column names that are dates (skip the first 3 columns)
date_columns = df.columns[3:]

# Convert the date strings to datetime objects and store in a list
dates = [datetime.strptime(date, '%Y-%m-%d') for date in date_columns]

print("Reading the network") # Adding some comments because it is a long process
#ps_network = PSNetwork(dates, "/path/to/xml/files")
ps_network = PSNetwork(dates, "./topo", "./triangulation_results.csv", "./psc_phases.csv")

parameter_estimator = NetworkParameterEstimator(ps_network)
print("Start parameter estimation") # Adding some comments because it is a long process
params = parameter_estimator.estimate_network_parameters()
print("Save parameters") # Adding some comments because it is a long process
save_network_parameters(params, ps_network, './psc_results.h5')



