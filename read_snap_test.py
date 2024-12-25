import sys
import numpy as np

#import jpy
sys.path.append('/home/timo/.snap/snap-python')
#java_home_dir = '/usr/lib/jvm/java-21-openjdk-amd64/lib/server'
#import jpy
import esa_snappy

# Read the .dim file
product = esa_snappy.ProductIO.readProduct('/home/timo/Data/LVS1_snap/deburst/S1A_IW_SLC__1SDV_20230702T134404_20230702T134432_049245_05EBEA_A4DF_Orb_Stack_esd_deb.dim')

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

