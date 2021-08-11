# Generic Propensity Modeling notebooks

This folder contains Python notebook templates for end-to-end propensity
modelling. These templates rely on
[gps_building_blocks](https://github.com/google/gps_building_blocks) library
to build step-by-step executable notebooks.

1. Data Audit and Exploratory Data Analysis.
2. Preparing ML ready dataset.
3. Splitting and balancing ML datasets.
4. Creating ML propensity model.
5. Model Evaluation and Diagnostics.
6. Scoring.
7. Experiment design and audience generation.
8. Audience upload.
9. Post-campaign analysis.

## Description of templates


### 4. Splitting and balancing ML datasets.


#### Overview

Before we train Machine Learning models we need to prepare appropriate
datasets for training, validation and testing workflows.
We also might want to make other
adjustments (for example deal with missing values, outliers and class imbalance).
This module provides utilities and templates for handling those situation.

#### Objectives

1. Filter and select relevant data.
2. Provide data balancing strategies and create balanced dataset.
3. Provide data splitting strategies and create separate datasets for TRAIN/VALIDATION/TEST.

#### Requirements

