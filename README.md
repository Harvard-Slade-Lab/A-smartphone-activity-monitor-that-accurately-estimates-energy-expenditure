# A smartphone activity monitor that accurately estimates energy expenditure

This repository contains the code and data for the paper "A smartphone activity monitor that accurately estimates energy expenditure". The project involves training and validating a data-driven model for estimating energy expenditure using smartphone during physical activity.

## Repository Structure

The repository is organized into three main folders:

1. **Training data-driven energy expenditure estimation model**:
   - This folder contains code and data for training a data-driven energy expenditure model using cross-validation on a training dataset from a previous study.
   - Running `plot_feat_importance.py` will generate and save the feature importance ranking of the trained data-driven model (gradient-boosted trees).

2. **Visualization of real-world walking experiment results**:
   - This folder includes scripts for loading, processing, and visualizing data from real-world walking experiments using OpenMetabolics and other activity monitors including pedometer, heart rate model, and smartwatch.
   - The `validation_dataset` folder contains dataset of 28 participants.
   - For OpenMetabolics' estimation, the code performs gait segmentation by peak detection, discretizes segmented data, extracts statistical features, and estimates energy expenditure using a pre-trained data-driven estimation model.
   - The code generates raw energy expenditure estimate results from various methods and saves them in the `indiv_plot` folder.
   - The overall results of the real-world walking experiments are saved as a box plot in the `validation_dataset` folder.

3. **OpenMetabolics' pipeline code for a week-long monitoring study**:
   - This folder provides the complete pipeline code for OpenMetabolics, which estimates energy expenditure when a smartphone is carried in a pocket during a week-long monitoring test.

Each folder includes the necessary data, code, and results for validating OpenMetabolics. To run the code in each folder, execute `main.py`.

## Note
We will release the OpenMetabolics smartphone application on the app store and as open-source code upon publication.

## Installation

To run the code provided in this repository, ensure you have the following packages installed with the specified versions:

```bash
pip install matplotlib==3.10.0 numpy==2.2.3 pandas==1.3.5 scikit-learn==1.0.2 scipy==1.15.2 xgboost==1.7.6