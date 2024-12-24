import sys
import numpy as np

#import jpy
sys.path.append('/home/timo/.snap/snap-python')
#java_home_dir = '/usr/lib/jvm/java-21-openjdk-amd64/lib/server'
#import jpy
import esa_snappy

# Read the .dim file
product = esa_snappy.ProductIO.readProduct('/home/timo/Data/LVS1_snap/topo/S1A_IW_SLC__1SDV_20230328T134359_20230328T134427_047845_05BFAE_DD75_Orb_Stack_Ifg_Deb.dim')

# Get the width and height
width = product.getSceneRasterWidth()
height = product.getSceneRasterHeight()

# Convert band to numpy array

band_data = np.zeros(width * height, dtype=np.float32)
band.readPixels(0, 0, width, height, band_data)
band_data = band_data.reshape(height, width)

