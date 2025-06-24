import pandas as pd
import plotly.express as px




# Load the dataset
file_path = "Dataset_VisContest_Rapid_Alloy_development_v3.txt"
df = pd.read_csv(file_path, sep='\t')




# Define columns for plotting
columns_to_plot = [
    'Al',
    'Si',
    'Cu',
    'Mg',
    'Fe',
    'delta_T',
    'eut. frac.[%]',
    'YS(MPa)',
    'hardness(Vickers)',
    'Therm.conductivity(W/(mK))',
    'Therm. diffusivity(m2/s)'
]




# Drop rows with missing values
df_filtered = df[columns_to_plot + ['CSC']].dropna()




# Clip CSC values to [0, 1] for consistent coloring
df_filtered['CSC_clipped'] = df_filtered['CSC'].clip(0, 1)




# Create the parallel coordinates plot using the clipped CSC values
fig = px.parallel_coordinates(
    df_filtered,
    dimensions=columns_to_plot,
    color="CSC_clipped",
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=0.5,
    range_color=[0, 1]
)




fig.show()
