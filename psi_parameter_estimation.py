import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, List, Dict, Union
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path


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
        self.temporal_baselines = temporal_baselines
        self.perpendicular_baselines = perpendicular_baselines
        self.range_distances = range_distances
        self.incidence_angles = incidence_angles

    def estimate_parameters_along_edge(self,
                                       phase_differences: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Estimate height error and velocity along network edge

        Parameters:
        -----------
        phase_differences: np.ndarray
            Wrapped phase differences along edge for each interferogram

        Returns:
        --------
        height_error: float
            Estimated height error
        velocity: float
            Estimated linear velocity
        residuals: np.ndarray
            Phase residuals after parameter estimation
        """

        def model_phase(params: List[float],
                        n_images: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Model phase components based on height and velocity

            Parameters:
            -----------
            params: List[float]
                [height_error, velocity, wrapping_factors]
            n_images: int
                Number of interferograms

            Returns:
            --------
            total_phase: np.ndarray
                Modeled total phase
            topo_phase: np.ndarray
                Topographic phase component
            motion_phase: np.ndarray
                Motion phase component
            """
            height_error = params[0]
            velocity = params[1]
            wrapping_factors = params[2:2 + n_images]

            # Topographic phase component
            topo_phase = (4 * np.pi * height_error * self.perpendicular_baselines /
                          (self.wavelength * self.range_distances * np.sin(self.incidence_angles)))

            # Motion phase component
            motion_phase = (4 * np.pi * velocity * self.temporal_baselines / self.wavelength)

            # Add wrapping factors
            total_phase = topo_phase + motion_phase + 2 * np.pi * wrapping_factors

            return total_phase, topo_phase, motion_phase

        def residual_function(params: List[float]) -> np.ndarray:
            """
            Calculate residuals between modeled and observed phases
            """
            modeled_phase, _, _ = model_phase(params, len(phase_differences))
            residuals = np.angle(np.exp(1j * (phase_differences - modeled_phase)))
            return residuals

        # Initial parameter guess
        n_images = len(phase_differences)
        initial_params = np.zeros(2 + n_images)  # [height_error, velocity, wrapping_factors]

        # Bounds for parameters
        # Height error bounds can be adjusted based on expected DEM error
        # Velocity bounds based on expected deformation rates
        # Wrapping factors must be integers
        bounds_lower = [-100, -0.5] + [-5] * n_images  # Example bounds
        bounds_upper = [100, 0.5] + [5] * n_images

        # Solve using least squares with integer constraints for wrapping factors
        result = least_squares(residual_function,
                               initial_params,
                               bounds=(bounds_lower, bounds_upper),
                               method='trf')

        # Extract results
        height_error = result.x[0]
        velocity = result.x[1]

        # Calculate final residuals
        _, topo_phase, motion_phase = model_phase(result.x, n_images)
        residuals = phase_differences - (topo_phase + motion_phase)
        residuals = np.angle(np.exp(1j * residuals))

        return height_error, velocity, residuals


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
            'residuals': {}
        }

        for edge_id, edge_data in self.network['edges'].items():
            height_error, velocity, residuals = (
                self.parameter_estimator.estimate_parameters_along_edge(
                    edge_data['phase_differences']
                )
            )

            network_parameters['height_errors'][edge_id] = height_error
            network_parameters['velocities'][edge_id] = velocity
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
        """Process triangle and points files to create edge data"""
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

            # Get phase data for each point
            p1_phases = points_df.loc[p1, self.dates[0].strftime('%Y-%m-%d'):self.dates[-1].strftime('%Y-%m-%d')]
            p2_phases = points_df.loc[p2, self.dates[0].strftime('%Y-%m-%d'):self.dates[-1].strftime('%Y-%m-%d')]
            p3_phases = points_df.loc[p3, self.dates[0].strftime('%Y-%m-%d'):self.dates[-1].strftime('%Y-%m-%d')]

            # Create edges (3 edges per triangle)
            # Edge 1: p1 -> p2
            self._edges[edge_counter] = {
                'start_point': p1,
                'end_point': p2,
                'phase_differences': p1_phases.values - p2_phases.values
            }
            edge_counter += 1

            # Edge 2: p2 -> p3
            self._edges[edge_counter] = {
                'start_point': p2,
                'end_point': p3,
                'phase_differences': p2_phases.values - p3_phases.values
            }
            edge_counter += 1

            # Edge 3: p3 -> p1
            self._edges[edge_counter] = {
                'start_point': p3,
                'end_point': p1,
                'phase_differences': p3_phases.values - p1_phases.values
            }
            edge_counter += 1

    def _process_xml_files(self):
        # ... (keep existing _process_xml_files implementation)
        pass

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
df = pd.read_csv('/home/timo/Data/LasVegasDesc/aps_psc_phases.csv')

# Get the column names that are dates (skip the first 3 columns)
date_columns = df.columns[3:]

# Convert the date strings to datetime objects and store in a list
dates = [datetime.strptime(date, '%Y-%m-%d') for date in date_columns]

#ps_network = PSNetwork(dates, "/path/to/xml/files")
ps_network = PSNetwork(dates, "/home/timo/Data/LasVegasDesc/topo")

parameter_estimator = NetworkParameterEstimator(ps_network)

print('Test')
