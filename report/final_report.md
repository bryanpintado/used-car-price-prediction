# Predicting Used Car Prices with Machine Learning

**GitHub Repository:** https://github.com/bryanpintado/used-car-price-prediction

---

# Abstract

Looking at how machines learn to guess what a used car might cost, based on facts like who made it, what fuel it uses, gears, engine details, miles covered. Built from records showing make, power source, shift method, motor capacity, distance traveled. Before any number crunching started, gaps in info got filled, messy bits cleaned up, labels turned into numbers. One approach tried straight-line math, another grouped decisions through many tiny trees - both tested side by side. Errors measured by average gap, bigger mistakes weighted more heavily, plus how much variation was actually explained. Out in the tests, Random Forest pulled ahead of Linear Regression by a clear margin - R squared landed near 0.89. That kind of jump shows how grouping models together can dig deeper into car price patterns when fed actual market data.

---

# 1. Introduction

Predicting used car prices is an important problem within the automotive market because vehicle pricing depends on many interacting variables including mileage, model year, fuel type, engine characteristics, and transmission type. Accurate pricing models can help buyers, sellers, and dealerships make more informed decisions.

The purpose of this project is to develop machine learning models capable of estimating vehicle prices using structured automotive data. The project follows a complete data science workflow including:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model training
- Performance evaluation
- Visualization and interpretation of results

Two regression-based machine learning models were implemented:
1. Linear Regression
2. Random Forest Regressor

The primary objective was to determine which model produced the most accurate price predictions while maintaining a reproducible and interpretable machine learning pipeline.

---

# 2. Related Work

Most times you will find machine learning shaping how prices get predicted - think homes, stocks, cars. Because these setups link number inputs with category traits, regression tools show up a lot. What happens is patterns form where past data guides future guesses. One reason those models stick around? They handle mixed data without falling apart. Prices shift, data changes, yet the method keeps tracking.

Most times, Linear Regression serves as a starting point because it stays clear and straightforward. Still, plenty of actual data follows curved patterns - something straight-line methods miss entirely.

What happens when you stack tree upon tree? The model starts seeing deeper shapes in data, not just noise. Picture a crowd guessing together - each voice small, yet the whole somehow wiser. One tree might drift off track, but many trees balance each other out. Instead of locking onto random quirks in numbers, they spread risk across branches. Patterns emerge clearer that way. Structure matters - and these forests thrive where rows and columns line up neatly. Accuracy climbs without tipping into obsession over tiny details.

From the start, one method follows a straight path while the other combines many twisting routes to see which guesses better. Instead of assuming simplicity, it tests how curves stack up against lines when forecasting outcomes.

---

# 3. Methodology

## 3.1 Dataset

The dataset used in this project contains 5,847 observations and multiple features related to used vehicles. The dataset includes:
- Vehicle manufacturer/model
- Location
- Year
- Kilometers driven
- Fuel type
- Transmission type
- Owner type
- Mileage
- Engine size
- Power
- Number of seats
- Vehicle price

The target variable used for prediction was:
- Price

---

## 3.2 Data Cleaning and Preprocessing

Several preprocessing steps were performed before model training.

### Removed Columns

The following columns were removed:
- `Unnamed: 0` because it only represented row indexing
- `New_Price` because it contained excessive missing values

### Cleaning Numerical Text Features

Several columns contained numerical values stored as text with measurement units:
- Mileage values contained strings such as "18.9 kmpl"
- Engine values contained strings such as "1197 CC"
- Power values contained strings such as "82 bhp"

Regular expressions were used to extract numerical values and convert these columns into floating-point data types.

### Missing Value Handling

Missing numerical values were replaced using median imputation. Missing categorical values were replaced using the most frequent category.

### Feature Encoding

Categorical variables were transformed using One-Hot Encoding through a preprocessing pipeline implemented with Scikit-learn.

### Train/Test Split

The dataset was divided into:
- 80% training data
- 20% testing data

A fixed random state was used to ensure reproducibility.

---

# 4. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to better understand feature distributions and relationships within the dataset before model training.

Several important trends were identified:
- Newer vehicles generally had higher prices
- Higher mileage and kilometers driven were associated with lower prices
- Engine power and engine size showed positive correlation with vehicle price
- Certain fuel types appeared more frequently than others

The exploratory analysis included several visualizations:
- Price distribution histogram
- Scatter plot of Year vs Price
- Scatter plot of Kilometers Driven vs Price
- Fuel type distribution chart
- Correlation heatmap

These visualizations helped identify relationships between variables and guided model selection.

---

# 5. Model Training

Two machine learning models were implemented and compared.

## 5.1 Linear Regression

Linear Regression was used as a baseline model because it provides a simple and interpretable relationship between features and target values.

## 5.2 Random Forest Regressor

Random Forest Regression was implemented using 100 decision trees. This ensemble-based approach was expected to better capture nonlinear relationships present within the dataset.

Both models were trained using preprocessing pipelines to ensure consistent transformations during training and prediction.

---

# 6. Experiments and Results

Models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 3.81 | 5.85 | 0.71 |
| Random Forest | 1.46 | 3.51 | 0.89 |

The Random Forest model significantly outperformed Linear Regression across all evaluation metrics.

Additional visualizations were generated to analyze model performance, including:
- Model comparison bar chart
- Actual versus predicted price plot
- Random Forest feature importance chart

Feature importance analysis revealed that:
- Vehicle year
- Power
- Kilometers driven
- Engine size

were among the most influential predictors of used car prices.

---

# 7. Conclusion

This project demonstrated how machine learning techniques can be used to predict used car prices using structured automotive datasets.

The project involved:
- Data preprocessing
- Feature engineering
- Exploratory Data Analysis
- Machine learning model development
- Performance evaluation

The Random Forest model achieved the best overall performance with an R² score of approximately 0.89, indicating strong predictive capability.

The results suggest that ensemble-based methods are better suited for modeling complex pricing relationships in automotive datasets compared to traditional linear models.

---

# 8. Future Work

Potential future improvements include:
- Hyperparameter optimization
- Cross-validation
- Additional feature engineering
- Integration of external market data
- Deep learning approaches for structured tabular data

---

# References

1. Scikit-learn Documentation: https://scikit-learn.org/

2. Pandas Documentation: https://pandas.pydata.org/

3. Matplotlib Documentation: https://matplotlib.org/

4. Seaborn Documentation: https://seaborn.pydata.org/