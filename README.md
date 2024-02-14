# Credit-Card-Default-Prediction
Internship at iNeuron

## Project Overview
This project entails building a classification model to predict credit card defaulters using a common dataset. It aims to forecast the likelihood of credit card defaults in the next month based on customers' demographic information and their behavior over the last six months.

## Motivation
Financial challenges like job loss, medical emergencies, or business downturns can lead to unmanageable debt situations, particularly with credit cards due to high interest rates and penalties. This model addresses the need to predict potential credit defaulters by analyzing demographic data (such as gender, age, marital status) and behavioral data (such as payment history, past transactions) to mitigate risks for banks.

## Dataset Information
The dataset encompasses credit card clients' default payments, demographic details, credit information, payment history, and bill statements in Taiwan from April 2005 to September 2005.

## Technical Aspects
This project involves two main components:
1. Training a RandomForestClassifier classification model to accurately predict defaulters.
    - Data cleaning and feature engineering
    - Application of machine learning classification model
2. Developing and deploying a Flask web application on Heroku.
    - Creation of the web app using Flask API
    - GitHub repository upload
    - Retrieval of customer information from the web app
    - Display of predictions

## Installation
The code is written in Python 3.7. If Python is not installed, it can be downloaded from [here](https://www.python.org/downloads/). For users with older Python versions, upgrading to the latest version via pip is recommended. To install required packages and libraries, execute the following command in the project directory after cloning the repository:
```bash
pip install -r requirements.txt
```

## Directory Structure
```
├── templates 
│   └── index.html
├── app.py
├── credit-card-default.csv
├── credit_default_prediction.py
├── model.pkl
├── Procfile
├── README.md
├── HLD document
├── LLD Document
├── Detailed Description Presentation
├── log file
├── wireframe pdf
├── README.md
└── requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://numpy.org/images/logos/numpy.svg" width=100>](https://numpy.org)    [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/450px-Pandas_logo.svg.png" width=150>](https://pandas.pydata.org)    [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=150>](https://scikit-learn.org/stable)   [<img target="_blank" src="https://www.statsmodels.org/stable/_images/statsmodels-logo-v2-horizontal.svg" width=170>](https://www.statsmodels.org)

[<img target="_blank" src="https://matplotlib.org/_static/logo2_compressed.svg" width=170>](https://matplotlib.org)      [<img target="_blank" src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width=150>](https://seaborn.pydata.org)

[<img target="_blank" src="https://jupyter.org/assets/nav_logo.svg" width=150>](https://jupyter.org)

## Team
Shikha Pandey: [GitHub](https://github.com/Shikha-Pandey)

## Credits
- The dataset has been provided by [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset). The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) at the UCI Machine Learning Repository. This project wouldn't have been possible without this dataset.