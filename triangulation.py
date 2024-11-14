import pandas as pd
import numpy as np
from scipy.spatial import Delaunay

# Read the input CSV file, assuming first unnamed column contains IDs
df = pd.read_csv('input.csv')
# Rename the unnamed first column to 'point_id'
df = df.rename(columns={df.columns[0]: 'point_id'})

# Extract coordinates
points = df[['sample', 'line']].values

# Perform Delaunay triangulation
tri = Delaunay(points)

# Create a list to store the triangulation results
triangles = []
for simplex in tri.simplices:
    # Get the point IDs for each triangle
    triangle_data = {
        'triangle_id': len(triangles),
        'point1_id': df['point_id'].iloc[simplex[0]],
        'point2_id': df['point_id'].iloc[simplex[1]],
        'point3_id': df['point_id'].iloc[simplex[2]]
    }
    triangles.append(triangle_data)

# Create DataFrame with triangulation results
results_df = pd.DataFrame(triangles)

# Save to CSV
results_df.to_csv('triangulation_results.csv', index=False)