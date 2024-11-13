import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

# Read the complex SAR data using rasterio
#with rasterio.open('path_to_your_complex_sar.tiff') as src:
with rasterio.open('/home/timo/Data/LasVegasDesc/resampled/TSX-1_0_2010-09-19.tiff') as src:
    # Read the complex data
    complex_data = src.read(1)
    # Get the transform for correct coordinate mapping
    transform = src.transform

# Calculate amplitude from complex data
amplitude = np.abs(complex_data)

# Read the CSV file with points
#points_df = pd.read_csv('path_to_your_points.csv')
points_df = pd.read_csv('/home/timo/Data/LasVegasDesc/aps_psc.csv')

# Create the visualization
plt.figure(figsize=(12, 8))

# Plot the amplitude image
# Using log scale for better visualization of SAR amplitude
plt.imshow(10 * np.log10(amplitude + 1e-10),
           cmap='gray',
           extent=[transform[2],
                  transform[2] + transform[0] * amplitude.shape[1],
                  transform[5] + transform[4] * amplitude.shape[0],
                  transform[5]])

# Overlay the points from CSV
# Assuming 'line' and 'sample' columns represent pixel coordinates
plt.scatter(points_df['sample'], points_df['line'],
           c='red',
           s=0.5, #changed manually to be smaller
           marker='o',
           label='Points')

plt.colorbar(label='Amplitude [dB]')
plt.title('SAR Image Amplitude with Points Overlay')
plt.xlabel('Range [pixels]')
plt.ylabel('Azimuth [pixels]')
plt.legend()
plt.grid(True)
plt.show()