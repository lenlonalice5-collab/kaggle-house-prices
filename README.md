# House Price Prediction

Machine Learning Course Design Project based on the Kaggle House Prices dataset.

## Project Introduction

This project aims to predict house sale prices using machine learning algorithms. The dataset is provided by Kaggle's House Prices competition.

The project uses Python and Scikit-Learn to build a regression model and generate predictions for unseen test data.

## Dataset

Source:

* Kaggle House Prices Competition

Files:

* train.csv
* test.csv
* sample_submission.csv

Target Variable:

* SalePrice

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn

## Machine Learning Workflow

1. Data Loading
2. Feature Selection
3. Missing Value Processing
4. Model Training
5. Prediction
6. Submission File Generation

## Selected Features

The following features are used for prediction:

* OverallQual
* GrLivArea
* GarageCars
* TotalBsmtSF
* YearBuilt

## Model

Random Forest Regressor

Parameters:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

## Project Structure

```text
house-price-prediction/
│
├── train.csv
├── test.csv
├── sample_submission.csv
├── predict.py
├── my_submission.csv
└── README.md
```

## Installation

Install required packages:

```bash
pip install pandas numpy scikit-learn
```

## Run

```bash
python predict.py
```

After execution, the prediction file will be generated:

```text
my_submission.csv
```

## Results

The generated submission file can be uploaded to Kaggle for evaluation and leaderboard ranking.

## Author

Machine Learning Course Design Project
Jimei University
