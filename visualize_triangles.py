import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import numpy.ma as ma

# Read the input files
# SAR image

#with rasterio.open('sar_image.tiff') as src:
with rasterio.open('') as src:
    # Read complex data
    sar_data = src.read(1)
    # Calculate amplitude
    amplitude = np.abs(sar_data)

    # Get transformation parameters
    transform = src.transform

# Read points and triangles
#points_df = pd.read_csv('points.csv')
points_df = pd.read_csv('')
# Rename the unnamed first column to 'point_id'
points_df = points_df.rename(columns={points_df.columns[0]: 'point_id'})

#triangles_df = pd.read_csv('triangulation_results.csv')
triangles_df = pd.read_csv('')

# Create the plot
plt.figure(figsize=(12, 8))

plt.imshow(amplitude, #10 * np.log10(amplitude + 1e-10),
           cmap='gray',
           vmin=0, vmax=200.0)
           #extent=[transform[2],
           #       transform[2] + transform[0] * amplitude.shape[1],
           #       transform[5] + transform[4] * amplitude.shape[0],
           #       transform[5]])

#Timo: to ensure compatibility between the displays for publication, I changed to the peviously used imshow
# # Plot SAR amplitude in dB scale with proper colormap
# amplitude_db = 20 * np.log10(amplitude + 1e-10) #manually added +1e-10 based on the debugging info Claude gave last time
# # Mask potential invalid values (zeros or very small values)
# amplitude_db = ma.masked_where(amplitude_db < -50, amplitude_db)
#
# plt.imshow(amplitude_db,
#            cmap='gray',
#            aspect='equal',
#            extent=[0, amplitude.shape[1], amplitude.shape[0], 0])  # Correct image orientation

# Plot triangles
for _, triangle in triangles_df.iterrows():
    # Get points for each triangle
    point1 = points_df[points_df['point_id'] == triangle['point1_id']]
    point2 = points_df[points_df['point_id'] == triangle['point2_id']]
    point3 = points_df[points_df['point_id'] == triangle['point3_id']]

    # Draw lines between points
    plt.plot([point1['sample'].values[0], point2['sample'].values[0]],
             [point1['line'].values[0], point2['line'].values[0]],
             'r-', linewidth=0.5, alpha=0.7)
    plt.plot([point2['sample'].values[0], point3['sample'].values[0]],
             [point2['line'].values[0], point3['line'].values[0]],
             'r-', linewidth=0.5, alpha=0.7)
    plt.plot([point3['sample'].values[0], point1['sample'].values[0]],
             [point3['line'].values[0], point1['line'].values[0]],
             'r-', linewidth=0.5, alpha=0.7)

#plt.colorbar(label='Amplitude [dB]')
plt.title('SAR Amplitude with Triangulation Overlay')
plt.xlabel('Sample')
plt.ylabel('Line')

# Adjust plot limits to match the image extent
plt.xlim(0, amplitude.shape[1])
plt.ylim(amplitude.shape[0], 0)  # Reverse y-axis to match image coordinates

plt.tight_layout()
plt.savefig('/home/timo/Data/LasVegasDesc/las_vegas_triangles.png', dpi=150, bbox_inches='tight')
#plt.show()