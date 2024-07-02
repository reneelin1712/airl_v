import pandas as pd

# Load the CSV file
df = pd.read_csv('trajectories_all_actions_rewards.csv')

# Calculate average real reward for each Next State
next_state_rewards = df.groupby('Next State')['Real Reward'].mean().reset_index()
next_state_rewards.to_csv('average_real_rewards.csv', index=False)

# Prepare data for the second output file
# Extracting all action next states and their corresponding rewards
action_data = pd.DataFrame({
    'Next State': df[['Action 1 Next State', 'Action 2 Next State', 'Action 3 Next State']].values.flatten(),
    'Reward': df[['Action 1 Reward', 'Action 2 Reward', 'Action 3 Reward']].values.flatten()
})

# Drop NaN values from the DataFrame
action_data = action_data.dropna(subset=['Next State', 'Reward'])

# Group by 'Next State' and calculate the average reward
action_rewards = action_data.groupby('Next State')['Reward'].mean().reset_index()
action_rewards.to_csv('average_action_rewards.csv', index=False)
