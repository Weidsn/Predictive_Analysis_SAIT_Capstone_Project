# Auto Insurance Churn Analysis

## Project Overview

This Repository records key components of the Capstone Project for the Data Analytics (DAAN) program at the Southern Alberta Institute of Technology (SAIT), Calgary, AB, which was completed in December 2023. 

The project, which was conducted in a group of two, by Sylvain and myself, focused on analyzing auto insurance churn. This repository contains key parts of the project, where we utilized Machine Learning techiques to predict customer churn on the basis of a number of customer features. 

## Documentation Contents

### 1. Data Dictionary

A list and description of the files relevant to this project can be found in this [Document](https://github.com/Weidsn/capstone_project/blob/main/Readme.txt). 

### 2. Data Wrangling

Raw data was taken from [Kaggle](https://www.kaggle.com/datasets/merishnasuwal/auto-insurance-churn-analysis-dataset?select=autoinsurance_churn.csv), which contains over 1.6 million records that were generated artificially.

The raw dataset was cleaned and prepared for analysis. Link to the code for [Data Cleaning](https://github.com/Weidsn/capstone_project/blob/main/data_cleaning_group2.py).

### 3. Training Predictive Models

A number of techniques were used to train models that aim to predict customer churn. Artificial Neural Networks (ANN), Decision Tree and Logisitic Regression were amoung the techniques employed.

Here is the code for training the [Artificial Neural Network (ANN)](https://github.com/Weidsn/capstone_project/blob/main/ChurnAnalysis.py) models. 

### 4. Evaluating the Models
During the training sessions, underperforming models were discarded, while promising one underwent a more detailed evaluation phase. 

Amongst the models evaluated are [ANN with Oversampling](https://github.com/Weidsn/capstone_project/blob/main/ann_resampled_group2.py), [ANN without Oversampling](https://github.com/Weidsn/capstone_project/blob/main/ann_origsample_group2.py), [Decision Tree](https://github.com/Weidsn/capstone_project/blob/main/decisiontree_group2.py), and [Logistic Regression](https://github.com/Weidsn/capstone_project/blob/main/regression_group2.py).

### 5. Final Presentation

Our insights and findings were summarized in this PowerPoint [Presentation](https://uofc-my.sharepoint.com/:p:/g/personal/weidong_sun1_ucalgary_ca/EWDtnpEmRShPs4EHnyqQYZQBcGMNAcHMwHbqxv8qGnve0Q?e=KDgaMd). 

We also presented a number of peculiar features we found in the raw data using Power BI [Dashboards](https://app.powerbi.com/view?r=eyJrIjoiMDljZDNlMDEtOWMwOC00NDc4LTk0YmMtNGVlMTQ5NzdhODFkIiwidCI6ImY1MmYyMTgzLTlmNjctNGFkMi1iNjU2LTZmNzU0ZmUxOTZjYiIsImMiOjZ9). 
