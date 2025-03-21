# Short Term Load Forecasting using ANN

This repository contains code for implementing Short Term Load Forecasting using Artificial Neural Networks (ANN). The dataset used for training and testing is `Copy_of_dataset.txt`, which includes features related to global power consumption and timestamps.

## Steps to Implement:

1. **Load and Preprocess Data**: Read the dataset, clean missing values, and extract relevant features such as date, time, and power consumption values.
2. **Feature Engineering and Normalization**: Extract time-based features (hour, day, month) and normalize data to standardize the input values.
3. **Split Data into Training and Testing Sets**: Use an 80-20 split to separate data for model training and evaluation.
4. **Define and Configure Neural Network**: Create a feedforward neural network with 20 hidden neurons and configure training parameters.
5. **Train the Network**: Train the neural network using MATLAB’s `fitnet` function with specified train-validation-test ratios.
6. **Evaluate the Model**: Predict values on test data, denormalize results, and compute performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
7. **Visualizations**: Generate plots including the neural network diagram, training performance graph, regression plots, error histograms, and time series responses.

## Files Included:

- `Copy_of_dataset.txt`: Original dataset used for training and testing.
- `main.m`: MATLAB script containing the complete code for Short Term Load Forecasting using ANN.
- `README.md`: This file, providing an overview of the project and instructions.

## Results:

The project includes visualizations such as network diagrams, performance graphs, error histograms, regression plots, time series responses, and error autocorrelation plots to analyze the forecasting performance.

## Dependencies:

Ensure MATLAB with Neural Network Toolbox is installed to run the code successfully.

## Usage:

1. Clone the repository.
2. Open `main.m` in MATLAB.
3. Run the script to load data, train the ANN, and evaluate the forecasting performance.

Feel free to explore and modify the code as needed for your specific forecasting tasks.
