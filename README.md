# Predicting Used Car Prices with Machine Learning

## Project Overview

This project explores the use of machine learning models to predict used car prices based on vehicle attributes such as mileage, model year, fuel type, transmission, and other features.

The project follows a complete data science workflow including:

- Data cleaning
- Exploratory data analysis (EDA)
- Feature engineering
- Model training
- Model evaluation
- Visualization of results

## Dataset

The dataset used for this project is stored in:
data/raw/train.csv

## Repository Structure

- data/
  - raw/ : original dataset
  - processed/ : cleaned datasets
- notebooks/ : EDA and modeling notebooks
- src/ : preprocessing, training, and evaluation scripts
- figures/ : saved visualizations
- report/ : final project report

## Models Used

- Linear Regression
- Random Forest Regressor

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## How to Run

Install requirements:

pip install -r requirements.txt

Run notebook:

jupyter notebook
