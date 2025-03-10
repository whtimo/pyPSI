import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import esa_snappy

input_dim_file = ''

product = esa_snappy.ProductIO.readProduct(input_dim_file)

# Get dimensions
width = product.getSceneRasterWidth()
height = product.getSceneRasterHeight()

# Get intensity bands
band_names = product.getBandNames()
intensity_band = [name for name in band_names if name.startswith('Intensity') and '_mst_' in name]

if not intensity_band:
    raise ValueError("No intensity master band found in the product")

band = product.getBand(intensity_band[0])
line_data = np.zeros((height, width), dtype=np.float32)
band.readPixels(0, 0, width, height, line_data)

# Calculate amplitude from complex data
amplitude = np.sqrt(line_data)

# Read the CSV file with points
#points_df = pd.read_csv('path_to_your_points.csv')
points_df = pd.read_csv('/home/timo/Data/LVS1_snap/psc.csv')

# Create the visualization
plt.figure(figsize=(12, 8))

# Plot the amplitude image
# Using log scale for better visualization of SAR amplitude
plt.imshow(amplitude, #10 * np.log10(amplitude + 1e-10),
           cmap='gray',
           vmin=-0, vmax=256)
           #extent=[transform[2],
           #       transform[2] + transform[0] * amplitude.shape[1],
           #       transform[5] + transform[4] * amplitude.shape[0],
           #       transform[5]])

#Overlay the points from CSV
#Assuming 'line' and 'sample' columns represent pixel coordinates
plt.scatter(points_df['sample'], points_df['line'],
           c='red',
           s=0.5, #changed manually to be smaller
           marker='o',
           label='Points')

#plt.colorbar(label='Amplitude [dB]')
plt.title('SAR Image Amplitude with PSC Overlay')
plt.xlabel('Range [pixels]')
plt.ylabel('Azimuth [pixels]')
plt.legend()
#plt.grid(True)
plt.savefig('', dpi=150, bbox_inches='tight')
#plt.show()