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

def denormalize_feature(normalized_feature, feature_max, feature_min):
    # Convert from [-1, 1] range to [0, 1] range
    feature_01 = (normalized_feature + 1) / 2
    
    # Denormalize the feature
    original_feature = feature_01 * (feature_max - feature_min) + feature_min
    original_feature_int = np.round(original_feature).astype(int)

    return original_feature_int

def evaluate_rewards(test_traj, policy_net, discrim_net, env, path_max, path_min):
    device = torch.device('cpu')  # Use CPU device
    policy_net.to(device)  # Move policy_net to CPU
    discrim_net.to(device)  # Move discrim_net to CPU

    reward_data = []
    input_features = []
    path_features = []
    denormalized_path_features = []

    for episode_idx, episode in enumerate(test_traj):
        des = torch.LongTensor([episode[-1].next_state]).long().to(device)
        for step_idx, x in enumerate(episode):
            state = torch.LongTensor([x.cur_state]).to(device)
            next_state = torch.LongTensor([x.next_state]).to(device)
            action = torch.LongTensor([x.action]).to(device)
            
            # Get the input features
            neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, next_path_feature, next_edge_feature = discrim_net.get_input_features(state, des, action, next_state)
            
            # Get the log probability of the action
            log_prob = policy_net.get_log_prob(state, des, action)
            
            # Calculate the reward using the discriminator
            reward = discrim_net.forward_with_actual_features(neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action, log_prob, next_path_feature, next_edge_feature)
            
            # Denormalize path feature
            path_feature_np = path_feature.detach().cpu().numpy().flatten()
            denormalized_path_feature = denormalize_feature(path_feature_np, path_max, path_min)
            
            reward_data.append({
                'episode': episode_idx,
                'step': step_idx,
                'state': x.cur_state,
                'action': x.action,
                'next_state': x.next_state,
                'reward': reward.item(),
                'path_feature': str(path_feature_np.tolist()),
                'denormalized_path_feature': str(denormalized_path_feature.tolist())
            })
            
            # Save path feature separately
            path_features.append(path_feature_np)
            denormalized_path_features.append(denormalized_path_feature)
            
            # Append other features
            input_features.append(np.concatenate((
                neigh_path_feature.detach().cpu().numpy().flatten(),
                neigh_edge_feature.detach().cpu().numpy().flatten(),
                edge_feature.detach().cpu().numpy().flatten(),
                action.detach().cpu().numpy().flatten(),
                log_prob.detach().cpu().numpy().flatten(),
                next_path_feature.detach().cpu().numpy().flatten(),
                next_edge_feature.detach().cpu().numpy().flatten()
            )))
    
    # Convert reward_data to a pandas DataFrame
    reward_df = pd.DataFrame(reward_data)

    return input_features, path_features, denormalized_path_features, reward_df

def save_to_csv(data, path_features, denormalized_path_features, filename):
    """Utility function to save data to a CSV file with separate path features."""
    if isinstance(data, pd.DataFrame):
        # If it's already a DataFrame, just add the path_features and denormalized_path_features columns
        data['path_features'] = [str(feat) for feat in path_features]
        data['denormalized_path_features'] = [str(feat) for feat in denormalized_path_features]
        data.to_csv(filename, index=False)
        print(f"Saved DataFrame to {filename}")
    else:
        # If it's raw data, create a DataFrame with input_features, path_features, and denormalized_path_features
        df = pd.DataFrame({
            'input_features': [str(feat) for feat in data],
            'path_features': [str(feat) for feat in path_features],
            'denormalized_path_features': [str(feat) for feat in denormalized_path_features]
        })
        df.to_csv(filename, index=False)
        print(f"Saved raw data to {filename}")

# Load the model
load_model(model_p)

# Evaluate rewards
test_trajs = env.import_demonstrations_step(test_p)
input_features, path_features, denormalized_path_features, reward_df = evaluate_rewards(test_trajs, policy_net, discrim_net, env, path_max, path_min)

# Saving the reward dataframe and input features to CSV files
reward_csv_path = "./output/reward_data_reverse.csv"
features_csv_path = "./output/input_features_with_path.csv"

save_to_csv(reward_df, path_features, denormalized_path_features, reward_csv_path)
save_to_csv(input_features, path_features, denormalized_path_features, features_csv_path)