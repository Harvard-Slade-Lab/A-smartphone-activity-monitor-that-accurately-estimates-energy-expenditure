"""
Copyright (c) 2025 Harvard Ability Lab
Title: "A smartphone activity monitor that accurately estimates energy expenditure"
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import norm
import utils

# Load subject information from a CSV file
subj_csv = pd.read_csv('./subject_info.csv')
target_subj = 'S1'
subj_info = {
    'code': subj_csv.loc[subj_csv['subject'] == target_subj, 'subject'].values[0],
    'weight': subj_csv.loc[subj_csv['subject'] == target_subj, 'weight'].values[0],
    'height': subj_csv.loc[subj_csv['subject'] == target_subj, 'height'].values[0],
    'gender': subj_csv.loc[subj_csv['subject'] == target_subj, 'gender'].values[0],
    'age': subj_csv.loc[subj_csv['subject'] == target_subj, 'age'].values[0]
}

# Compute the basal metabolic 
stand_aug_fact = 1.41  # Standing augmentation factor
height = subj_info['height']
weight = subj_info['weight']
age = subj_info['age']
gender = subj_info['gender']
cur_basal = utils.basalEst(height, weight, age, gender, stand_aug_fact, kcalPerDay2Watt=0.048426)

# Load energy expenditure estimation and pocket motion correction models
data_driven_model = pickle.load(open('./model_weight/data_driven_ee_model.pkl', 'rb'))
pocket_motion_correction_model = pickle.load(open('./model_weight/pocket_motion_correction_model.pkl', 'rb'))

# Load daily smartphone data
csv_path = './example_data/daily_sp_pocket_data.csv'
df_sp = pd.read_csv(csv_path).values

# Constants for the low-pass filter
sampling_freq = 50  # Sampling frequency in Hz
cutoff_freq = 6  # Crossover frequency for low-pass filter in Hz
filt_order = 4  # Filter order

# Constants for bout detection algorithm
sliding_win = 200  # Window size for sliding window in samples (4 seconds at 50Hz)
gyro_norm_thres = 0.5  # Threshold for gyro norm in rad/s

# Define low-pass filter parameters
b, a = signal.butter(filt_order, cutoff_freq, btype='low', fs=sampling_freq)

# Filter the collected smartphone data
time_sp = df_sp[:, 0]  # Time data
gyro_sp = df_sp[:, 1:4]  # Gyroscope data
acc_sp = df_sp[:, 4:]  # Accelerometer data

# Initialize lists to collect energy expenditure data
ee_all = []
time_all = []

start_idx = 0
n_samples = len(gyro_sp)

while start_idx + sliding_win <= n_samples:
    # Step 1: Process each detected active bout
    cur_window_time = time_sp[start_idx:start_idx+sliding_win]
    cur_pocket_gyro = gyro_sp[start_idx:start_idx+sliding_win, :]
    cur_pocket_acc = acc_sp[start_idx:start_idx+sliding_win, :]

    # Calculate the L2 norm of gyro data within the current window
    l2_norm_gyro = np.linalg.norm(cur_pocket_gyro)

    if l2_norm_gyro > gyro_norm_thres:
        # Apply a low-pass filter to the current window signals
        cur_pocket_gyro = signal.filtfilt(b, a, cur_pocket_gyro, axis=0)
        cur_pocket_acc = signal.filtfilt(b, a, cur_pocket_acc, axis=0)

        # Step 2: Orientation alignment with the superior-inferior axis of the thigh
        opt_rotm_z_pocket, theta_z = utils.get_rotate_z(cur_pocket_acc)
        cur_pocket_gyro_rot_zx = np.matmul(cur_pocket_gyro, opt_rotm_z_pocket)

        # Find principal axis of angular velocity
        prin_idx_pocket = utils.find_prin_axis(cur_pocket_gyro_rot_zx)
        prin_gyro_pocket = cur_pocket_gyro_rot_zx[:, prin_idx_pocket]
        
        if np.abs(np.max(prin_gyro_pocket)) < np.abs(np.min(prin_gyro_pocket)):
            prin_gyro_pocket = -prin_gyro_pocket

        # Detect peaks in the principal angular velocity
        gait_peaks = utils.peak_detect(prin_gyro_pocket)
        
        if len(gait_peaks) > 1:  # Consider it a bout if more than 1 peak is detected
            gait_data_pocket = utils.segment_data(gait_peaks, cur_pocket_gyro_rot_zx, sliding_win)
            if len(gait_data_pocket) < 1: # if there is no data segmented
                continue

            # Step 2: Orientation alignment with the mediolateral axis of the thigh
            avg_gait_data_pocket = np.mean(gait_data_pocket, axis=0)
            opt_rotm_y_pocket, theta_y = utils.get_rotate_y(avg_gait_data_pocket, prin_idx_pocket)
            opt_rotm_pocket = np.matmul(opt_rotm_z_pocket, opt_rotm_y_pocket)
            cur_pocket_gyro_cal = np.matmul(cur_pocket_gyro, opt_rotm_pocket) 

            # Adjust rotation if necessary
            # If the observed positive peaks are smaller than negative peaks, rotate along y-axis of 180 degrees
            pos_idx = cur_pocket_gyro_cal[:, -1] > 0
            neg_idx = cur_pocket_gyro_cal[:, -1] < 0
            gyro_z_norm_pos = norm(cur_pocket_gyro_cal[pos_idx, -1], ord=2)
            gyro_z_norm_neg = norm(cur_pocket_gyro_cal[neg_idx, -1], ord=2)
            
            if gyro_z_norm_pos <= gyro_z_norm_neg:
                opt_rotm_pocket = np.matmul(opt_rotm_pocket, utils.rotm_y(np.pi))
                theta_y += 180
                if theta_y > 180:
                    theta_y -= 360
                cur_pocket_gyro_cal = np.matmul(cur_pocket_gyro, opt_rotm_pocket)

            # Step 3: Gait segmentation and energy expenditure estimation
            gait_peaks = utils.peak_detect(cur_pocket_gyro_cal[:, -1])

            ee_time, ee_est = utils.estimateMetabolics(
                model= data_driven_model, time= cur_window_time,
                gait_data = cur_pocket_gyro_cal, peak_index= gait_peaks,
                weight = weight, height= height, stride_detect_window=sliding_win, 
                correction_model=pocket_motion_correction_model
            )
            for cur_ee_time, cur_ee_est in zip(ee_time, ee_est):
                time_all.append(cur_ee_time)
                ee_all.append(cur_ee_est)
        else:
            # If it's not a detectable bout, assign basal rate
            time_all.append(np.median(cur_window_time))
            ee_all.append(cur_basal)
    else:
        # If no movement is detected, assign basal rate
        time_all.append(np.median(cur_window_time))
        ee_all.append(cur_basal)

    # Move to the next window
    start_idx += sliding_win

# Plot the energy expenditure over time
fig, ax = plt.subplots(figsize=[10, 4])

# Plot the energy expenditure data with a specific color and marker
plt.plot(time_all, ee_all, marker='o', linestyle='', alpha=0.5, color='dodgerblue', label='Estimated\nenergy expenditure')

# Add a horizontal grey dotted line representing the basal rate
plt.axhline(y=cur_basal, color='grey', linestyle='--', alpha=0.5, linewidth=2, label='Basal Rate')

# Add labels and configure plot aesthetics
plt.xlabel('Time (s)')
plt.ylabel('Energy expenditure (W)')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# Add legend to identify the basal rate line
plt.legend(frameon=False)

# Save the plot
output_dir = './example_plot'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'energy_expenditure_plot.png'),dpi=300)

# Display the plot
plt.show()

