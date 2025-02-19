# OpenMetabolics: Pipeline Code for One-Week Long Real-World Dataset While Carrying a Smartphone in a Pocket 

## overview
This repository contains the code for processing real-world inertial measurement unit (IMU) data collected from smartphones carried in pockets. It is designed for a week-long activity monitoring test conducted at home, aimed at monitoring daily energy expenditure in a natural setting.

## data description
The example dataset includes data from an IMU, with the following seven columns: 
1. **Time**: recorded in absolute seconds.
2. **Angular velocity**: measured over three axes (x, y, z) in radians per second (rad/s).
3. **Linear acceleration**: measured over three axes (x, y, z) in gravitational units (g).

Note: Due to the limited size of GitHub, we uploaded a certain portion of one day's data of one participant.

## Pipeline code
The energy expenditure estimation pipeline code is composed of seven key steps:
1. **Bout Detection**: Identifying active phases based on the norm of gyroscopic data.
2. **Orientation alignment**: Converting the phone's orientation to match that of the thigh.
3. **Gait segmentation**: Segmenting the data into individual gait cycles.
4. **Preprocessing**: Discretizing data from a single gait cycle.
5. **Motion artifact correction**: Correcting motion artifacts caused by the smartphone moving in a pocket.
6. **Statistical feature extraction**: Extracting statistical features from the smartphone data.
7. **Model feature merging and model input preparation**: Combining all extracted features and other data to prepare the input for a data-driven estimation model.

## usage
This is an offline processing code. To run the code, execute the `main.py` script:
```bash
python main.py