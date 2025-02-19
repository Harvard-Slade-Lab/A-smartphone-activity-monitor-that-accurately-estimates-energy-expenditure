"""
Copyright (c) 2024 Harvard Ability lab
Title: "A smartphone activity monitor that accurately estimates energy expenditure"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Read the feature importance data
df_f_i = pd.read_csv('./feature_importance/xgboost_model_features.csv').values.flatten()

# Extract feature values
subj_weight = df_f_i[0]
subj_height = df_f_i[1]
gait_dur = df_f_i[2]

# Extracting features and flattening them
g_x_feat = df_f_i[3:33]
g_y_feat = df_f_i[33:63]
g_z_feat = df_f_i[63:93]
g_x_stat = df_f_i[93:98]
g_y_stat = df_f_i[98:103]
g_z_stat = df_f_i[103:108]

# Labels for the features
x_label = [
    'Body weight (kg) (1)', 
    'Height (m) (1)', 
    'Gait duration (s) (1)', 
    r'$\omega_{x}$ (30)', 
    r'$\omega_{y}$ (30)', 
    r'$\omega_{z}$ (30)', 
    r'$\omega_{x}$ (5)', 
    r'$\omega_{y}$ (5)', 
    r'$\omega_{z}$ (5)'
]

# Summing feature importance for each feature group
g_x_feat_sum = g_x_feat.sum()
g_y_feat_sum = g_y_feat.sum()
g_z_feat_sum = g_z_feat.sum()
g_x_stat_sum = g_x_stat.sum()
g_y_stat_sum = g_y_stat.sum()
g_z_stat_sum = g_z_stat.sum()

# Data for plotting
data = np.array([
    subj_weight, 
    subj_height, 
    gait_dur, 
    g_x_feat_sum, 
    g_y_feat_sum, 
    g_z_feat_sum, 
    g_x_stat_sum, 
    g_y_stat_sum, 
    g_z_stat_sum
]) * 100

# Plot configuration
fontsize = 8
colors = ['lightgrey', 'lightgrey', 'dimgrey', 'dodgerblue', 'dodgerblue', 'dodgerblue', 'dodgerblue', 'dodgerblue', 'dodgerblue']
hatch_patterns = ['', '', '', '', '', '', '//', '//', '//']

# Set the font properties globally
plt.rcParams.update({'font.size': fontsize, 'font.family': 'Arial'})

# Create the plot
fig, ax = plt.subplots(figsize=(5.5, 3))
y_pos = np.arange(len(x_label))
bars = ax.barh(y_pos, data, color=colors, align='center', height=0.7)

# Apply hatch patterns to the last three bars
for bar, hatch in zip(bars, hatch_patterns):
    if hatch:
        bar.set_hatch(hatch)

# Configure the plot appearance
ax.set_yticks(y_pos)
ax.set_yticklabels(x_label)
ax.invert_yaxis()  # Highest bar at the top
ax.set_xlabel('Percentage of feature importance (%)', fontsize=fontsize)
ax.set_ylabel('Feature groups\n(number of features within group)', fontsize=fontsize, labelpad=20, va='center')
ax.set_xlim([0, 80])

# Add value labels to each bar
for bar, value in zip(bars, data):
    ax.text(value, bar.get_y() + bar.get_height()/2, f'{value:.1f}%', 
            va='center', ha='left', color='black', fontsize=fontsize)

# Remove top and right spines
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# Create custom legend
legend_elements = [
    Patch(facecolor='lightgrey', label='Subject-specific information'),
    Patch(facecolor='dimgrey', label='Gait-specific information'),
    Patch(facecolor='dodgerblue', label='Angular velocity'),
    Patch(facecolor='dodgerblue', hatch='//', label='Statistical features of\nangular velocity')
]
ax.legend(handles=legend_elements, frameon=False, loc='best')

# Save the plot
plt.tight_layout()
plt.savefig('./feature_importance/xgboost_feature_importance_rank.png', dpi=900)
# plt.show()
