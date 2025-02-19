# Training Dataset for a Data-Driven Energy Expenditure Model

This dataset is from the study:

Slade, Patrick, et al. "Sensing leg movement enhances wearable monitoring of energy expenditure." *Nature Communications*, 12.1 (2021): 4312.

The dataset comprises 36 subjects and 17 test conditions, totaling 265 unique conditions and 13,300 gait cycle data points. Below are the details of the test conditions conducted in a laboratory environment:

| Condition | Activity      | Speed/Intensity |
|-----------|---------------|-----------------|
| C02       | Walking       | 0.75 m/s        |
| C03       | Walking       | 1.25 m/s        |
| C04       | Walking       | 1.75 m/s        |
| C05       | Running       | 2.25 m/s        |
| C06       | Running       | 2.75 m/s        |
| C07       | Running       | 3.25 m/s        |
| C08       | Biking        | 20 Watts        |
| C11       | Biking        | 80 Watts        |
| C12       | Biking        | 150 Watts       |
| C14       | Walking       | 1.0 m/s         |
| C15       | Walking       | 1.5 m/s         |
| C16       | Running       | 2.5 m/s         |
| C17       | Running       | 3.0 m/s         |
| C18       | Step Climbing | 50 steps/min    |
| C19       | Step Climbing | 70 steps/min    |
| C20       | Biking        | 50 Watts        |
| C21       | Biking        | 120 Watts       |

Each folder contains `x.csv` and `y.csv` files. The `x.csv` file includes preprocessed inputs of size 108 by 1 for the data-driven model. Each row includes 110 elements, representing the following data:

- Age (years)
- Gender (0 for female, 1 for male)
- Body weight (kg)
- Height (m)
- Gait duration during the given gait cycle (seconds)
- 30 samples of segmented x-axis angular velocity (anterior-posterior of the thigh segment)
- 30 samples of segmented y-axis angular velocity (superior-inferior of the thigh segment)
- 30 samples of segmented z-axis angular velocity (mediolateral of the thigh segment)
- 5 statistical features for each x, y, and z axis of angular velocity

For training the data-driven energy expenditure model, we used 108 elements, excluding age and gender.

The `y.csv` file includes a single float value representing the steady-state metabolic rate during the given test condition, measured by a ground-truth respirometry system.

For more details, please refer to the OpenMetabolics' paper.