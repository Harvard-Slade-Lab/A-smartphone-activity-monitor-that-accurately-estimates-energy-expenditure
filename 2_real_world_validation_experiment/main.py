"""
Copyright (c) 2025 Harvard Ability lab
Title: "A smartphone activity monitor that accurately estimates energy expenditure"
"""

import os
import sys
import dataset as ds

# Define processing parameters
proc_params = {
    # Main data directory
    'val_data_path': './validation_dataset',
    
    # Target subjects for real-world walking experiment (n=28)
    'target_subj': [
        'S2', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S12', 'S13', 'S14',
        'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S23', 'S24', 'S25', 
        'S26', 'S27', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35'
    ],

    # Model weight directory
    'model_weight': './model_weight',
    'input_features': 3,
    'subj_data': 18,
    'num_bins': 30,
    'plot_data': True,

}

rwDataset_processing = ds.rwDataset(proc_params)  # real-world dataset
