Auto Insurance Churn Analysis

Prepared by 
Slyvain MBONGO & 
Weidong SUN
2023-12-17

===========================================================================
About this repository

0. requirements.txt ----------- List of python packages. For installation, (in terminal) pip install -r requirements.txt

1. regression.py ------- Training and evaluating a model built with Logistic Regression using SMOTE (oversampling)

2. decisiontree.py ----- Training and evaluating a model built with Decision Tree using SMOTE

3. churn_fnn_0.85_0.2_ep250_age65_0.395.pkl ---- Ready-to-use Artificial Neural Network (ANN) model trained without SMOTE (saved in binary pickle format)

4. churn_fnn_0.85_0.2_ep100_resampled_0.58.pkl - Ready-to-use ANN model trained with SMOTE (saved in binary pickle format)

5. ann_origsample.py -- Testing and evaluating the ANN model without SMOTE

6. ann_resampled.py --- Testing adn evaluating the ANN model with SMOTE

7. helper_model_evaluation.py ---------  Helper functions for plotting confusion matrix and roc curve

8. autoinsurance.csv  ---------- Raw dataset
https://www.kaggle.com/datasets/merishnasuwal/auto-insurance-churn-analysis-dataset?select=autoinsurance_churn.csv

9. data_cleaning.py ------------ Data Cleaning

===========================================================================
Data Dictinary

autoinsurance.csv and autoinsurance_cleaned.csv
---------------------------------------------------------------------------
INDIVIDUAL_ID -- Unique ID for a specific insurance customer

ADDRESS_ID -- Unique ID for the primary address associated with a customer

CURR_ANN_AMT -- The Annual dollar value paid by the customer. It is not the policy amount. It is actually amount the customer paid during the previous year.

DAYS_TENURE -- The time in days individual has been a customer with the insurance agency.

CUST_ORIG_DATE -- The date the individual became a customer.

AGE_IN_YEARS -- Age of the individual.

LATITUDE -- Lattitude of the address

LONGITUDE -- Longitude of the address

CITY -- City

STATE -- State

COUNTY -- County

HOME_MARKET_VALUE -- Estimate value of home

HOME_MARKET_VALUE_MIN -- the lower range of home market value

HOME_MARKET_VALUE_MAX -- the upper range

HOME_MARKET_VALUE_MID -- the average between upper and lower range

INCOME -- Estimated Income for the Household associated with the individual

HAS_CHILDREN -- Flag, 1 indicates the individual has children in the home, 0 otherwise.

LENGTH_OF_RESIDENCE -- Estimated number of years the individual has lived in their current home.

MARITAL_STATUS -- Estimated marital status. Married or Single.

HOME_OWNER -- Flag, 1 individual owns primary home, 0 otherwise.

COLLEGE_DEGREE -- Flag, 1 individual has a college degree or more, 0 otherwise.

GOOD_CREDIT -- Flag, 1 individual has FICO greater than 630, 0 otherwise.

ACCT_SUSPD_DATE -- Day of Account Suspension or Cancellation

CHURN -- Flag, 1 yes churn, 0 no churn
