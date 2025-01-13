import numpy as np
import matplotlib.pyplot as plt
import rasterio
import glob
import pandas as pd

# Read the CSV file
df = pd.read_csv('your_csv_file.csv')
dates = df.columns[3:].tolist()  # Get dates from columns, skipping first 3 columns

# Get the dates for the epochs we want to visualize
epochs_to_plot = [0, 5, 11, 17]
selected_dates = [dates[i] for i in epochs_to_plot]

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

# Find and sort the APS files
aps_files = sorted(glob.glob('*epoch_*.tif'))

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
        im = axs[idx].imshow(data, cmap='turbo', vmin=vmin, vmax=vmax)
        axs[idx].set_title(f'Date: {date}')
        axs[idx].axis('off')

# Add a single colorbar for all subplots
plt.colorbar(im, ax=axs, label='APS (rad)', orientation='horizontal',
             pad=0.01, aspect=40)

# Adjust layout
plt.tight_layout()
plt.show()
