# Validation Dataset and Visualization During Real-World Walking Experiments

## overview
This folder contains the dataset of 28 participants who performed real-world walking experiments. The participant's folder includes:

## data description
1. **Thigh motion data**: Captured using an IMU attached to the thigh, providing 3-axis angular velocity (rad/s) and linear acceleration (m/s^2).
2. **Smartwatch energy expenditure estimates**: Provided in Watts.
3. **Ground-truth respirometry data**: Recorded per breath and converted into Watts.
4. **Heart rate data**: Measured in beats per minute (bpm).
5. **Subject-specific data**: Includes basal rate (Watts), average resting metabolic rate from the respirometry (Watts), age (years), gender (Male or Female), body weight (kg), and height (m).

Each CSV file is loaded and processed to estimate energy expenditure using various energy expenditure estimation methods.

The validation results folder includes the overall absolute error of each activity monitor and result plots of individual participants (n=28).

## usage
To run the code, execute the `main.py` script:
```bash
python main.py