
import numpy as np
import esa_snappy

# Read the .dim file
product = esa_snappy.ProductIO.readProduct('')

# Get the width and height
width = product.getSceneRasterWidth()
height = product.getSceneRasterHeight()
names = product.getBandNames()
for name in names:
    print(name)

# Convert band to numpy array

#band_data = np.zeros(width * height, dtype=np.float32)
#band.readPixels(0, 0, width, height, band_data)
#band_data = band_data.reshape(height, width)

