"""
Copyright (c) 2025 Harvard Ability Lab
Title: "A smartphone activity monitor that accurately estimates energy expenditure"
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings
from scipy import signal
from numpy import linalg as LA  
import matplotlib.pyplot as plt  
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# Suppress specific warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class rwDataset:
    """
    Class for processing and analyzing real-world gait data.
    """

    def __init__(self, params):
        """
        Initialize the dataset object with parameters, load data, and preprocess it.
        
        Args:
            params (dict): Configuration parameters containing subject data paths and processing options.
        """
        self.params = params
        self.target_subj = self.params['target_subj']  # Target subject identifier
        self.val_data_path = self.params['val_data_path']
        self.preprocessing()  # Start preprocessing upon initialization

    def butter_filter(self, cutoff, fs, order, btype):
        """
        Design a Butterworth filter based on given specifications.

        Args:
            cutoff (float): Cutoff frequency of the filter.
            fs (float): Sampling rate of the data.
            order (int): Order of the filter.
            btype (str): Type of filter ('low', 'high', etc.).

        Returns:
            tuple: Filter coefficients (b, a).
        """
        nyq = 0.5 * fs  # Nyquist Frequency
        Wn = cutoff / nyq  # Normalized cutoff frequency
        b, a = butter(order, Wn, btype=btype)
        return b, a 

    def get_cum_met(self, met, time):
        """
        Calculate cumulative metabolic cost over time.

        Args:
            met (numpy array): Array of metabolic rates.
            time (numpy array): Array of time points.

        Returns:
            float: Cumulative metabolic cost.
        """
        return sum(met[idx + 1] * dt for idx, dt in enumerate(np.diff(time)))

    def processRawGait(self, data_array, start_ind, end_ind, age, gen, weight, height, num_bins=30):
        """
        Process and extract features from raw gait data.

        Args:
            data_array (numpy array): Complete dataset of gait data.
            start_ind (int): Starting index of the data segment.
            end_ind (int): Ending index of the data segment.
            age (float): Age of the subject.
            gen (str): Gender of the subject.
            weight (float): Weight of the subject.
            height (float): Height of the subject.
            num_bins (int): Number of bins for resampling the data.

        Returns:
            numpy array: Flattened feature array for model input.
        """
        # Extract gait cycle
        gait_data = data_array[start_ind:end_ind, :]
        dur_stride = gait_data.shape[0] / 100  # Duration of stride

        # Resample the gait data into fixed bins for uniform size
        bin_gait = signal.resample(gait_data, num_bins, axis=0)
        model_input = bin_gait.T.flatten()[:90]  # Flatten to 90 elements

        # Split data into gyro components
        gyro_x, gyro_y, gyro_z = model_input[:30], model_input[30:60], model_input[60:90]

        def get_features(signal):
            """
            Extract statistical features like mean, std, etc., from the signal.

            Args:
                signal (numpy array): Input signal data.

            Returns:
                numpy array: Extracted features from the signal.
            """
            from scipy.stats import skew
            mean_ = np.mean(signal)
            std_ = np.std(signal)
            med_ = np.median(signal)
            skew_ = skew(signal)
            pow_ = LA.norm(signal, ord=2)  # Euclidean norm for signal power
            return np.array([mean_, std_, med_, skew_, pow_])

        # Feature extraction
        gyro_x_feat = get_features(gyro_x)
        gyro_y_feat = get_features(gyro_y)
        gyro_z_feat = get_features(gyro_z)

        # Combine all features and additional attributes into a single array
        model_input = np.concatenate((
            model_input.reshape(-1, 1), 
            gyro_x_feat.reshape(-1, 1),
            gyro_y_feat.reshape(-1, 1),
            gyro_z_feat.reshape(-1, 1)
        ), axis=0).flatten()

        # Insert weight, height, and stride duration
        model_input = np.insert(model_input, 0, [weight, height, dur_stride])
        return model_input

    def estimateHeartrate(self, hr_val, weight, age, gender):
        """
        Estimate energy expenditure based on heart rate and subject details.

        Args:
            hr_val (float): Heart rate value.
            weight (float): Weight of the subject.
            age (float): Age of the subject.
            gender (str): Gender of the subject.

        Returns:
            float: Estimated energy expenditure in watts.
        """
        gender_code = 1 if gender == 'M' else 0  # Encode gender
        ee_est = (
            gender_code * (-55.0969 + 0.6309 * hr_val + 0.1988 * weight + 0.2017 * age) +
            (1 - gender_code) * (-20.4022 + 0.4472 * hr_val - 0.1263 * weight + 0.074 * age)
        )
        return ee_est * 16.6666666667  # Convert to watts

    def estimatePedometer(self, cad, gender, weight):
        """
        Estimate energy expenditure using cadence data from a pedometer.

        Args:
            cad (float): Cadence in steps per minute.
            gender (str): Gender of the subject.
            weight (float): Weight of the subject.

        Returns:
            float: Estimated energy expenditure.
        """
        gender_code = 1 if gender == 'M' else 0  # Encode gender
        MET = -7.065 + (0.105 * cad) if gender_code == 1 else -8.805 + (0.110 * cad)
        ee_est = MET / 1.162 * weight
        return max(ee_est, 0)  # Ensure non-negative energy expenditure

    def process_accelerometer_data(self, df_imu_data, epoch_duration=5, fs=100):
        """
        Process accelerometer data, segmenting it into epochs, then filtering and processing each epoch.

        Args:
            df_imu_data (DataFrame): DataFrame containing accelerometer data.
            epoch_duration (int): Duration of each epoch in seconds.
            fs (int): Sampling rate in Hz.

        Returns:
            tuple: Arrays for time and mean high-pass filtered vector magnitude.
        """
        x_time = df_imu_data['time (s)'].values
        x_data = df_imu_data[['acc_x (m/s^2)', 'acc_y (m/s^2)', 'acc_z (m/s^2)']].values

        samples_per_epoch = epoch_duration * fs  # Calculate samples per epoch
        time_thigh_hpfvm_list, mean_thigh_hpfvm_list = [], []

        # Iterate over data in increments of samples_per_epoch
        for start in range(0, len(x_data), samples_per_epoch):
            end = start + samples_per_epoch
            if end > len(x_data):
                break

            epoch_data = x_data[start:end, :] / 9.81  # Normalize acceleration
            thigh_vm = np.sqrt(np.sum(epoch_data**2, axis=1))  # Calculate vector magnitude

            # High-pass filter the data
            b_hpf, a_hpf = signal.butter(4, 0.2 / (fs / 2), btype='high', analog=False)
            thigh_hpfvm = signal.lfilter(b_hpf, a_hpf, thigh_vm) 

            thigh_hpfvm = np.multiply(1000.0, np.abs(thigh_hpfvm))  # Convert to milli-g
            mean_thigh_hpfvm = np.mean(thigh_hpfvm)
            mean_thigh_hpfvm_list.append(mean_thigh_hpfvm)

            current_time = np.median(x_time[start:end])
            time_thigh_hpfvm_list.append(current_time)

        return np.array(time_thigh_hpfvm_list), np.array(mean_thigh_hpfvm_list)

    def estimateAccelerometer(self, thigh_hpfvm_t, thigh_hpfvm):
        """
        Estimate energy expenditure from accelerometer data using a derived equation.

        Args:
            thigh_hpfvm_t (array): Time data points.
            thigh_hpfvm (array): High-pass filtered vector magnitude from accelerometer.

        Returns:
            tuple: Time points and estimated energy expenditure.
        """
        term = 20.3 + 0.6401 * thigh_hpfvm
        j_min_kg = (-1.25 + 1.1353 * term - 2.4281 * np.sqrt(term) - 0.00040270 * (term ** 2))
        w_kg = j_min_kg / 60  # Convert to watts per kg
        watts = w_kg * self.body_weight
        return thigh_hpfvm_t, watts

    def estimateMetabolics(self, est_model, df_imu_data):
        """
        Estimate metabolic cost from gait data using a pre-trained model.

        Args:
            est_model (model): Trained energy expenditure model.
            df_imu_data (DataFrame): IMU data.

        Returns:
            tuple: Arrays for time stamps and energy expenditure using both OpenMet and Pedometer models.
        """
        imu_time = df_imu_data['time (s)'].values
        imu_gyro = df_imu_data[['gyro_x (rad/s)', 'gyro_y (rad/s)', 'gyro_z (rad/s)']].values
        b, a = self.butter_filter(cutoff=6, fs=100, order=4, btype='low')

        n_samples = df_imu_data.shape[0]
        fs = 100  # Sampling rate
        peak_height_thresh = np.deg2rad(70)  # Peak detection threshold
        peak_min_dist = int(0.6 * fs)  # Minimum peak distance
        stride_detect_window = 4 * fs  # Window to detect valid strides

        bout_threshold = 0.5  # Threshold to consider activity
        window_size = 400  # 4-second window
        start_idx = 0

        ee_ped, ee_ped_time, ee_openMet, ee_openMet_time = [], [], [], []

        # Process gyro data in windows
        while start_idx + window_size <= n_samples:
            cur_window = imu_gyro[start_idx:start_idx + window_size, :]
            cur_window_time = imu_time[start_idx:start_idx + window_size]
            
            cur_window = signal.filtfilt(b, a, cur_window, axis=0)  # Low-pass filter

            l2_norm_gyro = np.linalg.norm(cur_window)
            if l2_norm_gyro > bout_threshold:
                peaks, _ = signal.find_peaks(cur_window[:, 2], height=peak_height_thresh, distance=peak_min_dist)
                if not peaks.size:
                    # Append basal rate when no peaks detected
                    ee_openMet_time.append(np.median(cur_window_time))
                    ee_openMet.append(self.basal_rate)
                    ee_ped_time.append(np.median(cur_window_time))
                    ee_ped.append(self.basal_rate)
                else:
                    for i in range(len(peaks) - 1):
                        onset_time, offset_time = cur_window_time[peaks[i]], cur_window_time[peaks[i + 1]]
                        stride_len = np.abs(offset_time - onset_time) * fs

                        if stride_len <= stride_detect_window:
                            cad = 2 / (offset_time - onset_time) * 60  # Calculate cadence
                            ped_met = self.estimatePedometer(cad, self.gender, self.body_weight)
                            
                            ee_ped_time.append(onset_time)
                            ee_ped.append(ped_met)

                            gait_start_idx, gait_stop_idx = peaks[i], peaks[i + 1]
                            cur_gait_data = self.processRawGait(
                                cur_window, gait_start_idx, gait_stop_idx, 
                                self.age, self.gender, self.body_weight, self.height)
                            
                            model_input = cur_gait_data.reshape(1, -1)
                            ee_est_openMet = self.data_driven_ee_model.predict(model_input)[0]

                            ee_openMet.append(ee_est_openMet)
                            ee_openMet_time.append(onset_time)
            else:
                ee_openMet_time.append(np.median(cur_window_time))
                ee_openMet.append(self.basal_rate)
                ee_ped_time.append(np.median(cur_window_time))
                ee_ped.append(self.basal_rate)
            
            start_idx += window_size

        return np.array(ee_openMet_time), np.array(ee_openMet), np.array(ee_ped_time), np.array(ee_ped)

    def preprocessing(self):
        """
        Preprocess data for each subject and compute absolute error for various metrics.
        This function also performs visualization if enabled in parameters.
        """
        abs_err_openmet_cum_list, abs_err_sw_cum_list = [], []
        abs_err_hr_cum_list, abs_err_ped_cum_list = [], []
        abs_err_thigh_acc_cum_list = []

        for cur_subj in self.target_subj:
            print(cur_subj)
            val_data_path = self.val_data_path

            # Load respirometry data
            df_respirometry = pd.read_csv(os.path.join(val_data_path, cur_subj, 'respirometry_met.csv'))
            cur_gt_met_time = df_respirometry['time (s)'].values
            cur_gt_met = df_respirometry['metabolics (W)'].values

            # Load thigh IMU data
            df_imu_data = pd.read_csv(os.path.join(val_data_path, cur_subj, 'imu_thigh.csv'))

            # Load subject-specific information
            df_subj_info = pd.read_csv(os.path.join(val_data_path, cur_subj, 'subject_spec_info.csv'))

            # Capture subject information from csv
            self.age = df_subj_info['age (y)'].values[0]
            self.gender = df_subj_info['gender'].values[0]
            self.body_weight = df_subj_info['weight (kg)'].values[0]
            self.height = df_subj_info['height (m)'].values[0]
            self.basal_rate = df_subj_info['basal rate (W)'].values[0]
            self.rest_met = df_subj_info['rest metabolics (W)'].values[0]

            # Load pre-trained OpenMetabolics energy expenditure model
            self.data_driven_ee_model = pickle.load(
                open(self.params['model_weight'] + '/data_driven_ee_model.pkl', 'rb')
            )
            # Estimate metabolics using IMU data
            openMet_time, openMet_ee, ped_time, ped_met = self.estimateMetabolics(df_imu_data, df_imu_data)

            # Load heart rate data
            df_hr_data = pd.read_csv(os.path.join(val_data_path, cur_subj, 'hr_data.csv'))
            hr_data = df_hr_data['hr_data (bpm)'].values
            hr_time = df_hr_data['time (s)'].values
            hr_met = self.estimateHeartrate(hr_data, self.body_weight, self.age, self.gender)

            # Load smartwatch data
            df_smartwatch = pd.read_csv(os.path.join(val_data_path, cur_subj, 'smartwatch_est.csv'))
            sw_time = df_smartwatch['time (s)'].values
            sw_met = df_smartwatch['energy_estimates (W)'].values

            # Interpolating smartwatch data to a fixed interval
            new_timestamps = np.arange(sw_time[0], sw_time[-1] + 1, 5)
            interpolator = interp1d(sw_time, sw_met, kind='linear')
            sw_met = interpolator(new_timestamps)
            sw_time = new_timestamps

            # Estimate energy expenditure from accelerometer data
            thigh_hpfvm_t, thigh_hpfvm = self.process_accelerometer_data(df_imu_data)
            thigh_acc_time, thigh_acc_met = self.estimateAccelerometer(thigh_hpfvm_t, thigh_hpfvm) 

            # Determine cumulative time range for calculations
            cum_time_i = cur_gt_met_time[0]
            cum_time_e = cur_gt_met_time[-1] - 180

            # Calculate cumulative metabolic cost
            cum_met = self.get_cum_met(cur_gt_met, cur_gt_met_time)
            cum_met = (cum_met - 180 * self.rest_met) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            # Calculate cumulative energy expenditure for each method
            cum_idx_openMet = (openMet_time > cum_time_i) & (openMet_time < cum_time_e)
            cum_time_openMet = openMet_time[cum_idx_openMet]
            cum_openMet = self.get_cum_met(openMet_ee, cum_time_openMet) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            cum_idx_sw = (sw_time > cum_time_i) & (sw_time < cum_time_e)
            cum_time_sw = sw_time[cum_idx_sw]
            cum_sw = self.get_cum_met(sw_met[cum_idx_sw], cum_time_sw) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            cum_idx_hr = (hr_time > cum_time_i) & (hr_time < cum_time_e)
            cum_time_hr = hr_time[cum_idx_hr]
            cum_hr = self.get_cum_met(hr_met[cum_idx_hr], cum_time_hr) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            cum_idx_ped = (ped_time > cum_time_i) & (ped_time < cum_time_e)
            cum_time_ped = ped_time[cum_idx_ped]
            cum_ped = self.get_cum_met(ped_met[cum_idx_ped], cum_time_ped) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            cum_idx_thigh_acc = (thigh_acc_time > cum_time_i) & (thigh_acc_time < cum_time_e)
            cum_time_thigh_acc = thigh_acc_time[cum_idx_thigh_acc]
            cum_thigh_acc = self.get_cum_met(thigh_acc_met[cum_idx_thigh_acc], cum_time_thigh_acc) / (cur_gt_met_time[-1] - cur_gt_met_time[0])

            # Calculate absolute errors for each method
            abs_err_openmet = np.round(np.abs(cum_met - cum_openMet) / cum_met * 100, 3)
            abs_err_sw_cum = np.round(np.abs(cum_met - cum_sw) / cum_met * 100, 3)
            abs_err_hr_cum = np.round(np.abs(cum_met - cum_hr) / cum_met * 100, 3)
            abs_err_ped_cum = np.round(np.abs(cum_met - cum_ped) / cum_met * 100, 3)
            abs_err_thigh_acc_cum = np.round(np.abs(cum_met - cum_thigh_acc) / cum_met * 100, 3)

            # Print absolute errors
            print("Absolute err (OpenMET cumulative):", abs_err_openmet)
            print("Absolute err (SmartWatch cumulative):", abs_err_sw_cum)
            print("Absolute err (HeartRate model cumulative):", abs_err_hr_cum)
            print("Absolute err (Pedometer cumulative):", abs_err_ped_cum)
            print("Absolute err (Thigh-based accelerometer cumulative):", abs_err_thigh_acc_cum)
            print("\n")

            # Append results to respective lists
            abs_err_openmet_cum_list.append(abs_err_openmet)
            abs_err_sw_cum_list.append(abs_err_sw_cum)
            abs_err_hr_cum_list.append(abs_err_hr_cum)
            abs_err_ped_cum_list.append(abs_err_ped_cum)
            abs_err_thigh_acc_cum_list.append(abs_err_thigh_acc_cum)

            # Plot results if enabled
            if self.params['plot_data']:
                fontsize = 7.5
                csfont = {'fontname': 'helvetica'}
                title_fig = cur_subj
                fig, ax = plt.subplots(1, 1, figsize=(8, 2.5))
                
                # Plot various energy expenditure estimates
                ax.plot(cur_gt_met_time, cur_gt_met, '-', linewidth=1, color='black', label='Respirometry', alpha=1)
                ax.plot(openMet_time, openMet_ee, '-', linewidth=1, color='dodgerblue', label='OpenMetabolics', alpha=0.8)
                ax.plot(hr_time, hr_met, '-', linewidth=1, color='orangered', label='Heart-rate model', alpha=0.8)
                ax.plot(ped_time, ped_met, '-', linewidth=1, color='forestgreen', label='Pedometer', alpha=0.8)
                ax.plot(thigh_acc_time, thigh_acc_met, '-', linewidth=1, color='gold', label='Thigh-based\nAccelerometer', alpha=0.8)
                ax.plot(sw_time, sw_met, '-', linewidth=1, color='violet', label='Smartwatch', alpha=0.8)
                
                # Highlight onset and offset
                ax.axvline(x=cum_time_i, color='grey', linestyle='--', label='Activity\nonset/offset', alpha=0.8)
                ax.axvline(x=cum_time_e, color='grey', linestyle='--', alpha=0.8)

                # Customize plot
                plt.xticks(fontsize=fontsize, **csfont)
                plt.yticks(fontsize=fontsize, **csfont)
                from matplotlib.font_manager import FontProperties
                font_properties = FontProperties(family='helvetica', size=fontsize-3)
                ax.legend(frameon=False, prop=font_properties, loc='upper right')
                ax.set_ylabel('Energy expenditure (W)', fontsize=fontsize, **csfont)
                ax.set_xlabel('Absolute time(s)', fontsize=fontsize, **csfont)
                ax.spines['top'].set_color('none')
                ax.spines['right'].set_color('none')
                plt.suptitle(f'Real-world walking: {cur_subj}', fontsize=fontsize, **csfont)

                plt.tight_layout()
                save_path = './validation_results/indiv_plot/'
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(save_path + title_fig + '.png', dpi=300)

        # Convert lists to NumPy arrays for mean calculation
        abs_err_openmet_cum_list = np.array(abs_err_openmet_cum_list)
        abs_err_sw_cum_list = np.array(abs_err_sw_cum_list)
        abs_err_hr_cum_list = np.array(abs_err_hr_cum_list)
        abs_err_ped_cum_list = np.array(abs_err_ped_cum_list)
        abs_err_thigh_acc_cum_list = np.array(abs_err_thigh_acc_cum_list)

        # Calculate and print mean of absolute errors
        mean_abs_err_openmet_cum = np.mean(abs_err_openmet_cum_list)
        mean_abs_err_sw_cum = np.mean(abs_err_sw_cum_list)
        mean_abs_err_hr_cum = np.mean(abs_err_hr_cum_list)
        mean_abs_err_ped_cum = np.mean(abs_err_ped_cum_list)
        mean_abs_err_thigh_acc_cum = np.mean(abs_err_thigh_acc_cum_list)

        print(f"Mean absolute error (OpenMET cumulative): {mean_abs_err_openmet_cum:.0f}%")
        print(f"Mean absolute error (SmartWatch cumulative): {mean_abs_err_sw_cum:.0f}%")
        print(f"Mean absolute error (HeartRate model cumulative): {mean_abs_err_hr_cum:.0f}%")
        print(f"Mean absolute error (Pedometer cumulative): {mean_abs_err_ped_cum:.0f}%")
        print(f"Mean absolute error (Thigh-based accelerometer cumulative): {mean_abs_err_thigh_acc_cum:.0f}%")

        if self.params['plot_data']:
            # Setup data and labels for box plot
            data = [
                abs_err_openmet_cum_list, abs_err_sw_cum_list, 
                abs_err_hr_cum_list, abs_err_ped_cum_list, 
                abs_err_thigh_acc_cum_list
            ]
            labels = ['OpenMetabolics', 'SmartWatch', 'Heart-rate\nmodel', 
                      'Pedometer', 'Thigh-based\naccelerometer']
            colors = ['dodgerblue', 'orangered', 'violet', 'forestgreen', 'gold']
            means = [
                mean_abs_err_openmet_cum, mean_abs_err_sw_cum, 
                mean_abs_err_hr_cum, mean_abs_err_ped_cum, 
                mean_abs_err_thigh_acc_cum
            ]

            bar_width = 0.3
            meanpointprops = dict(marker='s', markerfacecolor='black', markeredgecolor='black', markersize=3)
            fontsize = 7.5
            csfont = {'fontname': 'helvetica'}
            fig_size = (4, 3)

            # Create box plot
            fig, ax = plt.subplots()
            for i in range(len(data)):
                box = ax.boxplot(
                    data[i], positions=[i + 1], patch_artist=True, 
                    widths=bar_width, boxprops=dict(facecolor=colors[i], color='black', alpha=0.8),
                    meanprops=meanpointprops, medianprops=dict(color='k'),
                    whiskerprops=dict(linestyle='-', linewidth=1.5),
                    capprops=dict(color='black'), showfliers=False,
                    flierprops=dict(marker='o', color='black', alpha=0.5),
                    showmeans=True, notch=False
                )

                # Calculate and annotate mean values
                whisker_vals = [whisker.get_ydata()[1] for whisker in box['whiskers']]
                text_position = whisker_vals[1] + 2
                ax.text(i + 1, text_position, f'{means[i]:.0f}%', fontsize=fontsize, 
                        ha='center', va='bottom', color='black', **csfont)

            # Configure x-ticks and labels
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xticklabels(labels)
            ax.set_xlabel(f'Real-world\nwalking bouts\n$n$ = {len(abs_err_openmet_cum_list)}', fontsize=fontsize, **csfont)
            ax.set_ylabel('Absolute error (%)', fontsize=fontsize, **csfont)

            # Remove unnecessary spines
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')

            # Customize plot appearance
            plt.xticks(fontsize=fontsize, **csfont)
            plt.yticks(fontsize=fontsize, **csfont)

            # Adjust and save plot
            fig.set_size_inches(fig_size)
            plt.tight_layout()
            save_path = './validation_results/'
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(save_path + 'real_world_walking_results.png', dpi=300)