import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import rasterio

# Read the TIFF file
with rasterio.open('/home/timo/Projects/WuhanTSXAirport/shpcountsmall.tiff') as src:
    data = src.read(1)  # Read the first band

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Create the image plot with turbo colormap
im = ax.imshow(data, cmap='turbo')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Pixel Values', rotation=270, labelpad=15)

# Add title
plt.title('Brotherhood Pixels Distribution')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()