import pandas as pd

# Read the data from the file
df = pd.read_csv('edge.txt')

# Add 'time_step' and 'speed' columns
df['time_step'] = 0  # Initialize 'time_step' column
df['speed'] = df['length']  # Use 'length' as 'speed' for now

# Normalize the 'speed' column
df['speed'] = (df['speed'] - df['speed'].min()) / (df['speed'].max() - df['speed'].min())

# Create duplicated rows for each time_step
time_steps = range(1, 8)  # Define time steps from 1 to 7
df_final = pd.concat([df.assign(time_step=ts) for ts in time_steps], ignore_index=True)

# Save the DataFrame back to a new .txt file
df_final.to_csv('edge_updated.txt', index=False)
