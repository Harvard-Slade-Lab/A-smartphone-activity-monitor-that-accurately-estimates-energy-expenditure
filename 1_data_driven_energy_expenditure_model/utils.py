"""
Copyright (c) 2025 Harvard Ability lab
Title: "A smartphone activity monitor that accurately estimates energy expenditure"
"""

import os
import numpy as np
from scipy.stats import skew
from numpy.linalg import norm

# Function to convert 3D input data to 2D arrays
def convert_input_dim(x_data, y_data):
    num_samples = x_data.shape[0] * x_data.shape[1]
    x_data_arr = np.ones((num_samples, x_data.shape[2]))
    y_data_arr = np.ones((num_samples, 1))
    
    jdx = 0
    for idx in range(x_data.shape[0]):  # Iterate over subjects/conditions
        cur_y_data = y_data[idx]
        for cur_x_data in x_data[idx]:  # Iterate over each gait cycle
            x_data_arr[jdx] = cur_x_data
            y_data_arr[jdx] = cur_y_data
            jdx += 1
    
    return x_data_arr, y_data_arr

# Function to extract features from signal data
def get_features(signal):
    mean_ = np.mean(signal, axis=1).reshape(-1, 1)
    std_ = np.std(signal, axis=1).reshape(-1, 1)
    med_ = np.median(signal, axis=1).reshape(-1, 1)
    skew_ = skew(signal, axis=1).reshape(-1, 1)
    pow_ = norm(signal, ord=2, axis=1).reshape(-1, 1)
    features = np.concatenate((mean_, std_, med_, skew_, pow_), axis=1)
    return features

# Function to load data from specified directory and list of files
def load_data(data_dir, data_list):
    x_all = []
    y_all = []
    
    for cur_file in data_list:
        if cur_file == '.DS_Store':  # Skip unwanted system file
            continue

        x_path = os.path.join(data_dir, cur_file, 'x.csv')
        y_path = os.path.join(data_dir, cur_file, 'y.csv')

        x_data = np.loadtxt(x_path, delimiter=',').astype(float)
        y_data = np.loadtxt(y_path, delimiter=',').astype(float)

        # Extract subject-specific and feature data
        x_data_subj = x_data[:, :5]
        x_data = x_data[:, 5:95]  # Only use the first 90 features

        # Split data into gyro components
        gyro_x = x_data[:, :30]
        gyro_y = x_data[:, 30:60]
        gyro_z = x_data[:, 60:90]

        # Compute features for each gyro component
        gyro_x_feat = get_features(gyro_x)
        gyro_y_feat = get_features(gyro_y)
        gyro_z_feat = get_features(gyro_z)

        # Combine features into a single array
        x_data_combined = np.concatenate(
            (x_data_subj[:, 2:], x_data, gyro_x_feat, gyro_y_feat, gyro_z_feat), axis=1
        )

        x_all.append(x_data_combined)
        y_all.append(y_data)

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    x_all, y_all = convert_input_dim(x_all, y_all)

    return x_all, y_all

# Function to load dataset and return it in a dictionary
def load_dataset(data_dir, data_list):
    x_train, y_train = load_data(data_dir, data_list)
    dataset = {'x_train': x_train, 'y_train': y_train}
    return dataset
