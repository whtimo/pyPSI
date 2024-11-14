import unittest
import numpy as np
import rasterio
import tempfile
import os
from pathlib import Path
import shutil
from amplitude_dispersion import calculate_amplitude_dispersion  # Your main function


class TestAmplitudeDispersion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data directory and create synthetic SAR images"""
        # Create temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        cls.output_file = os.path.join(cls.test_dir, 'amplitude_dispersion.tif')

        # Create synthetic data parameters
        cls.rows, cls.cols = 100, 100
        cls.n_images = 20

        # Create profile for test images
        cls.profile = {
            'driver': 'GTiff',
            'dtype': 'complex64',
            'nodata': None,
            'width': cls.cols,
            'height': cls.rows,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': rasterio.transform.from_bounds(0, 0, 1, 1, cls.cols, cls.rows)
        }

        # Generate test images
        cls.create_test_images()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        shutil.rmtree(cls.test_dir)

    @classmethod
    def create_test_images(cls):
        """Create synthetic SAR images with known properties"""
        # Create stable point (PS candidate)
        ps_amplitude = 1.0
        ps_noise = 0.1

        # Create noisy point (non-PS)
        noisy_amplitude = 1.0
        noisy_noise = 0.5

        for i in range(cls.n_images):
            # Create complex image
            image = np.zeros((cls.rows, cls.cols), dtype=np.complex64)

            # Add stable point
            ps_real = ps_amplitude + np.random.normal(0, ps_noise)
            ps_imag = ps_amplitude + np.random.normal(0, ps_noise)
            image[25, 25] = complex(ps_real, ps_imag)

            # Add noisy point
            noisy_real = noisy_amplitude + np.random.normal(0, noisy_noise)
            noisy_imag = noisy_amplitude + np.random.normal(0, noisy_noise)
            image[75, 75] = complex(noisy_real, noisy_imag)

            # Save image
            filename = os.path.join(cls.test_dir, f'test_image_{i:03d}.tif')
            with rasterio.open(filename, 'w', **cls.profile) as dst:
                dst.write(image, 1)

    def test_output_dimensions(self):
        """Test if output dimensions match input dimensions"""
        stats = calculate_amplitude_dispersion(self.test_dir, self.output_file)

        with rasterio.open(self.output_file) as src:
            self.assertEqual(src.shape, (self.rows, self.cols))

    def test_ps_candidate_detection(self):
        """Test if PS candidates are correctly identified"""
        stats = calculate_amplitude_dispersion(self.test_dir, self.output_file)

        with rasterio.open(self.output_file) as src:
            da = src.read(1)

            # Test stable point (should have low DA)
            self.assertLess(da[25, 25], 0.25)

            # Test noisy point (should have high DA)
            self.assertGreater(da[75, 75], 0.25)

    def test_invalid_pixels(self):
        """Test handling of invalid pixels"""
        stats = calculate_amplitude_dispersion(self.test_dir, self.output_file)

        with rasterio.open(self.output_file) as src:
            da = src.read(1)

            # Test if zero amplitude pixels are marked as NaN
            self.assertTrue(np.isnan(da[0, 0]))

    def test_statistical_properties(self):
        """Test statistical properties of the output"""
        stats = calculate_amplitude_dispersion(self.test_dir, self.output_file)

        # Test if stats dictionary contains expected keys
        expected_keys = ['min_da', 'max_da', 'mean_da', 'median_da',
                         'ps_candidates', 'total_valid_pixels', 'processed_images']
        for key in expected_keys:
            self.assertIn(key, stats)

        # Test if number of processed images is correct
        self.assertEqual(stats['processed_images'], self.n_images)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Create test image with extreme values
        extreme_image = np.zeros((self.rows, self.cols), dtype=np.complex64)
        extreme_image[50, 50] = complex(1e-10, 1e-10)  # Very small values
        extreme_image[60, 60] = complex(1e10, 1e10)  # Very large values

        filename = os.path.join(self.test_dir, 'extreme_test.tif')
        with rasterio.open(filename, 'w', **self.profile) as dst:
            dst.write(extreme_image, 1)

        # Test if processing handles extreme values without numerical errors
        try:
            stats = calculate_amplitude_dispersion(self.test_dir, self.output_file)
            with rasterio.open(self.output_file) as src:
                da = src.read(1)
                self.assertFalse(np.any(np.isinf(da[~np.isnan(da)])))
        except Exception as e:
            self.fail(f"Processing failed with extreme values: {str(e)}")

    def test_geospatial_preservation(self):
        """Test if geospatial information is preserved"""
        stats = calculate_amplitude_dispersion(self.test_dir, self.output_file)

        # Compare input and output geospatial properties
        with rasterio.open(os.path.join(self.test_dir, 'test_image_000.tif')) as src_in:
            with rasterio.open(self.output_file) as src_out:
                self.assertEqual(src_in.crs, src_out.crs)
                self.assertEqual(src_in.transform, src_out.transform)


def validate_with_reference_data(test_output, reference_file):
    """
    Validate results against reference data (if available)

    Parameters:
    -----------
    test_output : str
        Path to the output file to validate
    reference_file : str
        Path to reference data file

    Returns:
    --------
    dict
        Validation metrics
    """
    with rasterio.open(test_output) as test:
        test_data = test.read(1)

    with rasterio.open(reference_file) as ref:
        ref_data = ref.read(1)

    # Calculate validation metrics
    valid_mask = ~(np.isnan(test_data) | np.isnan(ref_data))

    metrics = {
        'rmse': np.sqrt(np.mean((test_data[valid_mask] - ref_data[valid_mask]) ** 2)),
        'mae': np.mean(np.abs(test_data[valid_mask] - ref_data[valid_mask])),
        'correlation': np.corrcoef(test_data[valid_mask], ref_data[valid_mask])[0, 1]
    }

    return metrics


if __name__ == '__main__':
    unittest.main()