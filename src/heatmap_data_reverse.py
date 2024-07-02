import numpy as np
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
import torch.nn.functional as F
import torch
from torch import nn
import math
import time
from network_env import RoadWorld
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from utils.torch import to_device
from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj

import csv

torch.backends.cudnn.enabled = False

def denormalize_feature(normalized_feature, feature_max, feature_min):
    print('feature_max',feature_max)
    print('feature_min',feature_min)
    # Convert from [-1, 1] range to [0, 1] range
    feature_01 = (normalized_feature + 1) / 2
    
    # Denormalize the feature
    original_feature = feature_01 * (feature_max - feature_min) + feature_min
    
    return original_feature

if __name__ == '__main__':
    log_std = -0.0  # log std for the policy
    gamma = 0.99  # discount factor
    tau = 0.95  # gae
    l2_reg = 1e-3  # l2 regularization regression (not used in the model)
    learning_rate = 3e-4  # learning rate for both discriminator and generator
    clip_epsilon = 0.2  # clipping epsilon for PPO
    num_threads = 4  # number of threads for agent
    min_batch_size = 8192  # 8192  # minimal batch size per PPO update
    eval_batch_size = 8192  # 8192  # minimal batch size for evaluation
    log_interval = 10  # interval between training status logs
    save_mode_interval = 50  # interval between saving model
    max_grad_norm = 10  # max grad norm for ppo updates
    seed = 1  # random seed for parameter initialization
    epoch_disc = 1  # optimization epoch number for discriminator
    optim_epochs = 10  # optimization epoch number for PPO
    optim_batch_size = 64  # optimization batch size for PPO
    cv = 0  # cross validation process [0, 1, 2, 3, 4]
    size = 1000  # size of training data [100, 1000, 10000]
    max_iter_num = 2000  # maximal number of main iterations {100size: 1000, 1000size: 2000, 10000size: 3000}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    """environment"""
    edge_p = "../data/edge.txt"
    network_p = "../data/transit.npy"
    path_feature_p = "../data/feature_od.npy"
    train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    test_p = "../data/cross_validation/test_CV%d.csv" % cv
    # test_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)
    """inialize road environment"""
    od_list, od_dist = ini_od_dist(train_p)
    env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
    """load path-level and link-level feature"""
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, link_max, link_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)

    # Example of denormalizing a single path feature
    normalized_single_path_feature = np.array([-3.8775510e-01, -3.0396718e-01, -1.0000000e+00, -3.6000001e-01,
                                            -1.0000000e+00, -1.0000000e+00, -1.0000000e+00, -1.0000000e+00,
                                            -6.9230771e-01, -1.0000000e+00, -4.1666670e-10, -1.0000000e+00])

    original_single_path_feature = denormalize_feature(normalized_single_path_feature, path_max, path_min)

    print("Normalized path feature:")
    print(normalized_single_path_feature)
    print("\nDenormalized (original) path feature:")
    print(original_single_path_feature)

    import numpy as np

# def minmax_normalization(feature, xmax, xmin):
#     normalized_feature = (feature - xmin) / (xmax - xmin + 1e-8)
#     normalized_feature = 2 * normalized_feature - 1
#     return normalized_feature

feature_max = np.array([4.900000e+01, 3.162756e+03, 0.000000e+00, 2.500000e+01, 0.000000e+00,
                        1.900000e+01, 3.300000e+01, 3.000000e+00, 1.300000e+01, 0.000000e+00,
                        2.400000e+01, 1.000000e+00])
feature_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

# Calculate normalized max and min
normalized_max = minmax_normalization(feature_max, feature_max, feature_min)
normalized_min = minmax_normalization(feature_min, feature_max, feature_min)

print("Normalized Max:", normalized_max)
print("Normalized Min:", normalized_min)
