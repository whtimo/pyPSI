from snapista import Registry
import xarray as xr

# Read the product
reader = Registry.ReadOp()
reader.file = 'path/to/your/file.dim'
product = reader.execute()

# Convert to xarray dataset
ds = xr.open_dataset(product)

