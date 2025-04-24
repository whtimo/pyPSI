import numpy as np
import matplotlib.pyplot as plt
import rasterio
import glob
import pandas as pd

# Read the CSV file
df = pd.read_csv('./psc.csv')
dates = df.columns[3:].tolist()  # Get dates from columns, skipping first 3 columns

# Get the dates for the epochs we want to visualize
epochs_to_plot = [0, 5, 11, 17]
selected_dates = [dates[i] for i in epochs_to_plot]

# Create figure with space for colorbar on right
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.1])
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
       fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

# Find and sort the APS files
aps_files = sorted(glob.glob('directory/*epoch_*.tif'))

# Create a list to store the image data for colorbar normalization
all_data = []
for epoch in epochs_to_plot:
    with rasterio.open(aps_files[epoch]) as src:
        data = src.read(1)
        all_data.append(data)

# Calculate global min and max for consistent colorbar
vmin = min(data.min() for data in all_data)
vmax = max(data.max() for data in all_data)

# Plot each image
for idx, (epoch, date) in enumerate(zip(epochs_to_plot, selected_dates)):
    with rasterio.open(aps_files[epoch]) as src:
        data = src.read(1)
        im = axs[idx].imshow(data, cmap='Greys', vmin=vmin, vmax=vmax)
        axs[idx].set_title(f'{date}')
        axs[idx].axis('off')

# Add colorbar on the right
cbar_ax = fig.add_subplot(gs[:, 2])
plt.colorbar(im, cax=cbar_ax, label='Residual')

# Adjust layout
plt.tight_layout()
plt.savefig('./residuals.png', dpi=600, bbox_inches='tight')

#plt.show()