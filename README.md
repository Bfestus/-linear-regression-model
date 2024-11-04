# Linear Regression Model for Sales Prediction

## Overview
This project builds a simple linear regression model to predict sales based on TV marketing expenses. The model is developed using three different approaches: NumPy, Scikit-Learn, and Gradient Descent from scratch. Additionally, the performance of the linear regression model is compared with other algorithms such as Random Forest and Decision Trees.

## Table of Contents
1. [Dataset](#dataset)
2. [Approaches](#approaches)
    - [Linear Regression with NumPy](#linear-regression-with-numpy)
    - [Linear Regression with Scikit-Learn](#linear-regression-with-scikit-learn)
    - [Linear Regression using Gradient Descent](#linear-regression-using-gradient-descent)
3. [Model Comparison](#model-comparison)
4. [Installation](#installation)
5. [Usage](#usage)
6. [License](#license)

## Dataset
The dataset used for this project is a simple Kaggle dataset saved in the file `data/tvmarketing.csv`. It contains two fields:
- **TV**: TV marketing expenses
- **Sales**: Sales amount

## Approaches

### Linear Regression with NumPy
- This approach utilizes NumPy's `polyfit` function to compute the slope and intercept of the linear regression line.

### Linear Regression with Scikit-Learn
- In this approach, Scikit-Learn's `LinearRegression` class is used to fit the model. The dataset is split into training and testing sets for evaluation.

### Linear Regression using Gradient Descent
- A custom implementation of the gradient descent algorithm is developed to minimize the sum of squares cost function and find the optimal coefficients.

## Model Comparison
The performance of the linear regression model is compared with Random Forest and Decision Trees using Root Mean Square Error (RMSE) as the evaluation metric. The models are ranked from best to worst based on their RMSE values.

## Installation
To run this project, you'll need to install the required packages. You can do this using pip:

```bash
pip install pandas matplotlib scikit-learn
