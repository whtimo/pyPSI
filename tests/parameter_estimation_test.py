import unittest
import numpy as np
from ps_parameter_estimation_periodogram import PSIParameterEstimator  # Replace 'your_module' with the actual module name
import random

class TestPSIParameterEstimator(unittest.TestCase):

    def setUp(self):
        # Fix random seed for reproducibility
        np.random.seed(42)
        random.seed(42)

        # Random size for arrays between 20 and 30
        self.size = random.randint(20, 30)

        # Random wavelength between 3 to 6 cm (0.03 to 0.06 meters)
        self.wavelength = random.uniform(0.03, 0.06)

        # Random temporal baselines between 10-300 days
        self.temporal_baselines = np.random.uniform(10, 300, size=self.size)

        # Random perpendicular baselines between -500 to 500 meters
        self.perpendicular_baselines = np.random.uniform(-500, 500, size=self.size)

        # Random range distances, all similar between 700 to 800 km (700000 to 800000 meters)
        self.range_distances = np.full(self.size, random.uniform(700000, 800000))

        # Random incidence angles, all identical between 30-50 degrees, converted to radians
        self.incidence_angles_degrees = random.uniform(30, 50)
        self.incidence_angles = np.full(self.size, np.radians(self.incidence_angles_degrees))

        # Create an instance of PSIParameterEstimator
        self.estimator = PSIParameterEstimator(
            self.wavelength,
            self.temporal_baselines,
            self.perpendicular_baselines,
            self.range_distances,
            self.incidence_angles
        )

    def test_wavelength(self):
        self.assertAlmostEqual(self.estimator.wavelength, self.wavelength)

    def test_temporal_baselines_years(self):
        expected_temporal_baselines_years = self.temporal_baselines / 365.0
        np.testing.assert_array_almost_equal(
            self.estimator.temporal_baselines_years,
            expected_temporal_baselines_years
        )

    def test_perpendicular_baselines(self):
        np.testing.assert_array_almost_equal(
            self.estimator.perpendicular_baselines,
            self.perpendicular_baselines
        )

    def test_range_distances(self):
        np.testing.assert_array_almost_equal(
            self.estimator.range_distances,
            self.range_distances
        )

    def test_incidence_angles(self):
        np.testing.assert_array_almost_equal(
            self.estimator.incidence_angles,
            self.incidence_angles
        )

    def test_estimate_parameters(self):
        # Define true height and velocity
        height_true = random.uniform(0, 50)
        velocity_true = random.uniform(-20, 20)

        # Compute height_to_phase and velocity_to_phase
        height_to_phase = (4 * np.pi / self.estimator.wavelength) * \
                          (self.estimator.perpendicular_baselines /
                           (self.estimator.range_distances * np.sin(self.estimator.incidence_angles)))
        velocity_to_phase = (4 * np.pi / self.estimator.wavelength) * self.estimator.temporal_baselines_years

        # Compute phase_topo_sim and phase_motion_sim
        phase_topo_sim = np.angle(np.exp(1j * height_true * height_to_phase))
        phase_motion_sim = np.angle(np.exp(1j * (velocity_true / 1000.0) * velocity_to_phase))

        # Compute model_phase_sim
        model_phase_sim = np.angle(np.exp(1j * (phase_topo_sim + phase_motion_sim)))

        # Set phase_differences
        phase_differences = model_phase_sim

        # Call estimate_parameters
        best_height, best_velocity, max_coherence = self.estimator.estimate_parameters(phase_differences)

        # Check if best_height and best_velocity are close to height_true and velocity_true
        self.assertAlmostEqual(best_height, height_true, delta=0.5)
        self.assertAlmostEqual(best_velocity, velocity_true, delta=1.5)

        # Check if max_coherence is close to 1
        self.assertAlmostEqual(max_coherence, 1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()