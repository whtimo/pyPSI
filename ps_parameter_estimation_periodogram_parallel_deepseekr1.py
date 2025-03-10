import numpy as np
#from scipy.optimize import least_squares
from typing import Tuple, List, Dict, Union
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
#import h5py
import time
from multiprocessing import Pool


def save_point_data_to_csv(input_csv, output_csv, params):
    """
    Save point data to CSV including id, sample, line, height errors, velocities, and temporal coherence.

    Parameters:
    -----------
    input_csv : str
        Path to input CSV file containing sample and line data
    output_csv : str
        Path to output CSV file
    params : dict
        Dictionary containing 'height_errors', 'velocities', and 'temporal_coherences' arrays
    point_ids : array-like
        Array of point IDs for height errors and temporal coherence
    edge_ids : array-like
        Array of edge IDs for velocities
    """


    # Read the input CSV file
    df = pd.read_csv(input_csv)
    ids = df.iloc[:, 0]
    # Create arrays for the parameters
    height_errors_array = np.array([params['height_errors'][point_id] for point_id in ids])
    velocities_array = np.array([params['velocities'][point_id] for point_id in ids])
    temporal_coherences_array = np.array([params['temporal_coherences'][point_id] for point_id in ids])

    # Create a new dataframe with the required columns
    output_df = pd.DataFrame({
        'id': df.iloc[:, 0],
        'sample': df['sample'],
        'line': df['line'],
        'height_errors': height_errors_array,
        'velocities': velocities_array,
        'temporal_coherence': temporal_coherences_array
    })

    # Save to CSV file
    output_df.to_csv(output_csv, index=False)

def find_matching_point_index(reference_point_file, first_csv, second_csv):
    # Read the reference point ID from the text file
    with open(reference_point_file, 'r') as f:
        point_id = int(f.read().strip())

    # Read both CSV files using pandas

    df1 = pd.read_csv(first_csv)
    df2 = pd.read_csv(second_csv)

    # Get the sample and line coordinates for the reference point
    reference_row = df1.iloc[point_id]
    sample = reference_row['sample']
    line = reference_row['line']

    # Find the matching index in the second CSV file
    matching_index = df2[(df2['sample'] == sample) &
                         (df2['line'] == line)].index

    # Return the first matching index if found, otherwise None
    return matching_index[0] if len(matching_index) > 0 else None


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

    def estimate_parameters(self, phase_differences):
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

        # Vectorized computation
        H = height_search[:, np.newaxis, np.newaxis]  # Shape: (200, 1, 1)
        V = velocity_search[np.newaxis, :, np.newaxis]  # Shape: (1, 200, 1)

        # Compute all model phases at once
        phase_topo = H * height_to_phase  # Shape: (200, 1, n_baselines)
        phase_motion = (V / 1000) * velocity_to_phase  # Shape: (1, 200, n_baselines)
        total_phase = phase_topo + phase_motion  # Shape: (200, 200, n_baselines)

        # Compute coherence in vectorized form
        exp_diff = np.exp(1j * phase_differences)  # Shape: (n_baselines,)
        exp_model = np.exp(-1j * total_phase)  # Shape: (200, 200, n_baselines)
        coherence_values = np.mean(exp_diff * exp_model, axis=2)  # Shape: (200, 200)
        coherence_matrix = np.abs(coherence_values)

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

        return best_height, best_velocity, max_coherence


class ParameterEstimator:
    def __init__(self, ps_network):
        """
        Estimate parameters for entire PS network

        Parameters:
        -----------
        ps_network: dict
            Network information including edges and phase data
        """
        self.ps_info = ps_info
        self.points = ps_info['points']
        self._ps_network = ps_network #Timo: Deepseek did overlook this, added manually
        self.parameter_estimator = PSIParameterEstimator(
            wavelength=ps_network['wavelength'],
            temporal_baselines=ps_network['temporal_baselines'],
            perpendicular_baselines=ps_network['perpendicular_baselines'],
            range_distances=ps_network['range_distances'],
            incidence_angles=ps_network['incidence_angles']
        )

    def estimate_parameters(self, ref_point: int) -> dict:
        parameters = {
            'height_errors': {},
            'velocities': {},
            'temporal_coherences': {}
        }
        ref_phases = self.points.iloc[ref_point][3:].to_numpy()

        # Prepare arguments for parallel processing
        args_list = []
        for point_id in range(len(self.points)):
            if point_id == ref_point:
                continue  # Skip reference point
            phases = self.points.iloc[point_id][3:].to_numpy()
            phase_diff = np.angle(np.exp(1j * (ref_phases - phases)))
            args = (phase_diff, self._ps_network)  # Assume self.ps_network stores original data
            args_list.append((point_id, args))

        # Process points in parallel
        with Pool() as pool:
            results = pool.map(self._process_point_parallel, args_list)

        # Collect results
        for point_id, (h_err, vel, tc) in results:
            parameters['height_errors'][point_id] = h_err
            parameters['velocities'][point_id] = vel
            parameters['temporal_coherences'][point_id] = tc

        # Handle reference point
        parameters['height_errors'][ref_point] = 0.0
        parameters['velocities'][ref_point] = 0.0
        parameters['temporal_coherences'][ref_point] = 1.0
        return parameters

    def _process_point_parallel(self, args):
        point_id, (phase_diff, ps_network) = args
        estimator = PSIParameterEstimator(wavelength=ps_network['wavelength'],  #Timo: Another syntax error from DeepseekR1
            temporal_baselines=ps_network['temporal_baselines'],
            perpendicular_baselines=ps_network['perpendicular_baselines'],
            range_distances=ps_network['range_distances'],
            incidence_angles=ps_network['incidence_angles'])
        h_err, vel, tc = estimator.estimate_parameters(phase_diff)
        return point_id, (h_err, vel, tc)


class PSInfo:
    def __init__(self, dates: List[datetime], xml_path: str, points_file: str):
        self.dates = dates
        self.xml_path = Path(xml_path)
        self._wavelength = None
        self._temporal_baselines = None
        self._perpendicular_baselines = None
        self._range_distances = None
        self._incidence_angles = None


        # Store file paths
        self._points = pd.read_csv(points_file)

        # Process XML files
        self._process_xml_files()


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
        elif key == 'points':
            return self._points
        else:
            raise KeyError(f"Key {key} not found in PSNetwork")

if __name__ == "__main__":
    start_time = time.perf_counter()
    # Read the CSV file
    df_psc = pd.read_csv('')
    df_ps = pd.read_csv('')
    # Get the column names that are dates (skip the first 3 columns)
    date_columns = df_ps.columns[3:]

    # Convert the date strings to datetime objects and store in a list
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in date_columns]

    #ref_point = 0
    ref_point = find_matching_point_index('', '', '')
    #print("Reading the network") # Adding some comments because it is a long process
    #ps_network = PSNetwork(dates, "/path/to/xml/files")
    ps_info = PSInfo(dates, "",  "")

    parameter_estimator = ParameterEstimator(ps_info)
    print("Start parameter estimation") # Adding some comments because it is a long process
    params = parameter_estimator.estimate_parameters(ref_point)
    print("Save parameters") # Adding some comments because it is a long process
    save_point_data_to_csv("", "", params)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


