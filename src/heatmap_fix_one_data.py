from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
import time
import torch
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj
from network_env import RoadWorld
from utils.torch import to_device
import numpy as np
import pandas as pd
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN

import shap

def load_model(model_path):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Value Model loaded Successfully")
    discrim_net.load_state_dict(model_dict['Discrim'])
    print("Discrim Model loaded Successfully")

cv = 0  # cross validation process [0, 1, 2, 3, 4]
size = 1000  # size of training data [100, 1000, 10000]
gamma = 0.99  # discount factor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv

"""environment"""
edge_p = "../data/edge.txt"
network_p = "../data/transit.npy"
path_feature_p = "../data/feature_od.npy"
train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv
model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)

"""initialize road environment"""
od_list, od_dist = ini_od_dist(train_p)
env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
"""load path-level and link-level feature"""
path_feature, path_max, path_min = load_path_feature(path_feature_p)
edge_feature, link_max, link_min = load_link_feature(edge_p)
path_feature = minmax_normalization(path_feature, path_max, path_min)
path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
edge_feature = minmax_normalization(edge_feature, link_max, link_min)
edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

"""define actor and critic"""
policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                    path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                    env.pad_idx).to(device)
value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                env.state_action, path_feature_pad, edge_feature_pad,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                env.pad_idx).to(device)


import torch
import numpy as np
import pandas as pd

def calculate_reward_with_varying_path_feature(test_traj, policy_net, discrim_net, env):
    device = torch.device('cpu')
    policy_net.to(device)
    discrim_net.to(device)

    reward_data = []

    # Only use the first step of the first trajectory
    episode = test_traj[0]
    x = episode[0]
    
    des = torch.LongTensor([episode[-1].next_state]).long().to(device)
    state = torch.LongTensor([x.cur_state]).to(device)
    next_state = torch.LongTensor([x.next_state]).to(device)
    action = torch.LongTensor([x.action]).to(device)

    # Get the input features
    neigh_path_feature, neigh_edge_feature, original_path_feature, edge_feature, next_path_feature, next_edge_feature = discrim_net.get_input_features(state, des, action, next_state)

    # Get the log probability of the action
    log_prob = policy_net.get_log_prob(state, des, action)

    # Iterate through different values for the second element of path_feature
    for path_feature_second_value in np.arange(-1.0, 1.1, 0.1):
        # Create a new path_feature tensor with the modified second value
        path_feature = original_path_feature.clone()
        print('path_feature',path_feature)
        path_feature[1] = path_feature_second_value

        # Calculate the reward using the discriminator
        reward = discrim_net.forward_with_actual_features(
            neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, 
            action, log_prob, next_path_feature, next_edge_feature
        )

        reward_data.append({
            'path_feature_second_value': path_feature_second_value,
            'reward': reward.item(),
        })

    # Convert reward_data to a pandas DataFrame
    reward_df = pd.DataFrame(reward_data)

    return reward_df

# Load the model
load_model(model_p)

# Get the first trajectory
test_trajs = env.import_demonstrations_step(test_p)

# Calculate rewards for varying path_feature second values
reward_df = calculate_reward_with_varying_path_feature(test_trajs, policy_net, discrim_net, env)

# Print the results
print(reward_df)

# Optionally, save to CSV
reward_csv_path = "./output/reward_data_varying_path_feature.csv"
reward_df.to_csv(reward_csv_path, index=False)
print(f"Saved results to {reward_csv_path}")



# import torch
# import numpy as np

# def calculate_reward(path_feature_second_value, policy_net, discrim_net):
#     # Fixed values with correct dimensions
#     neigh_path_feature = torch.tensor([
#         [-2.2449e-01, -1.3079e-01, -1.0000e+00, -2.0000e-01, -1.0000e+00, -5.7895e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, 1.6667e-01, -1.0000e+00],
#         [-3.8776e-01, -3.0397e-01, -1.0000e+00, -3.6000e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [-3.4694e-01, -2.2956e-01, -1.0000e+00, -2.8000e-01, -1.0000e+00, -1.0000e+00, -9.3939e-01, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00]
#     ]).unsqueeze(0)

#     neigh_edge_feature = torch.tensor([
#         [-2.2449e-01, -1.3079e-01, -1.0000e+00, -2.0000e-01, -1.0000e+00, -5.7895e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, 1.6667e-01, -1.0000e+00],
#         [-3.8776e-01, -3.0397e-01, -1.0000e+00, -3.6000e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [-3.4694e-01, -2.2956e-01, -1.0000e+00, -2.8000e-01, -1.0000e+00, -1.0000e+00, -9.3939e-01, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00]
#     ]).unsqueeze(0)

#     path_feature = torch.tensor([
#         [-2.2449e-01, path_feature_second_value, -1.0000e+00, -2.0000e-01, -1.0000e+00, -5.7895e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, 1.6667e-01, -1.0000e+00],
#         [-3.8776e-01, -3.0397e-01, -1.0000e+00, -3.6000e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [-3.4694e-01, -2.2956e-01, -1.0000e+00, -2.8000e-01, -1.0000e+00, -1.0000e+00, -9.3939e-01, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00]
#     ]).unsqueeze(0)

#     edge_feature = torch.tensor([
#         [-2.2449e-01, -1.3079e-01, -1.0000e+00, -2.0000e-01, -1.0000e+00, -5.7895e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, 1.6667e-01, -1.0000e+00],
#         [-3.8776e-01, -3.0397e-01, -1.0000e+00, -3.6000e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [-3.4694e-01, -2.2956e-01, -1.0000e+00, -2.8000e-01, -1.0000e+00, -1.0000e+00, -9.3939e-01, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00]
#     ]).unsqueeze(0)

#     next_path_feature = torch.tensor([
#         [-2.2449e-01, -1.3079e-01, -1.0000e+00, -2.0000e-01, -1.0000e+00, -5.7895e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, 1.6667e-01, -1.0000e+00],
#         [-3.8776e-01, -3.0397e-01, -1.0000e+00, -3.6000e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [-3.4694e-01, -2.2956e-01, -1.0000e+00, -2.8000e-01, -1.0000e+00, -1.0000e+00, -9.3939e-01, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00]
#     ]).unsqueeze(0)

#     next_edge_feature = torch.tensor([
#         [-2.2449e-01, -1.3079e-01, -1.0000e+00, -2.0000e-01, -1.0000e+00, -5.7895e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, 1.6667e-01, -1.0000e+00],
#         [-3.8776e-01, -3.0397e-01, -1.0000e+00, -3.6000e-01, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [-3.4694e-01, -2.2956e-01, -1.0000e+00, -2.8000e-01, -1.0000e+00, -1.0000e+00, -9.3939e-01, -1.0000e+00, -6.9231e-01, -1.0000e+00, -4.1667e-10, -1.0000e+00]
#     ]).unsqueeze(0)

#     # Convert state, des, and action to tensors
#     state = torch.LongTensor([[144]])
#     des = torch.LongTensor([[392]])
#     action = torch.LongTensor([[1]])

#     # Calculate log_prob
#     log_prob = policy_net.get_log_prob(state, des, action)

#     # Calculate reward
#     reward = discrim_net.forward_with_actual_features(
#         neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, 
#         action, log_prob, next_path_feature, next_edge_feature
#     )

#     return reward.item()

# # Assuming policy_net and discrim_net are already defined and loaded

# rewards = []
# path_feature_values = np.arange(-1.0, 1.1, 0.1)

# for value in path_feature_values:
#     reward = calculate_reward(value, policy_net, discrim_net)
#     rewards.append((value, reward))

# # Print the results
# for value, reward in rewards:
#     print(f"Path feature value: {value:.1f}, Reward: {reward}")