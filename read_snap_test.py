import snappy
from snappy import ProductIO

# Read the .dim file
product = ProductIO.readProduct('path/to/your/file.dim')

# Get the band you want to work with
band = product.getBand('band_name')  # e.g., 'Amplitude_VV', 'Intensity_VH'

# Get the width and height
width = product.getSceneRasterWidth()
height = product.getSceneRasterHeight()

# Convert band to numpy array
import numpy as np
band_data = np.zeros(width * height, dtype=np.float32)
band.readPixels(0, 0, width, height, band_data)
band_data = band_data.reshape(height, width)

