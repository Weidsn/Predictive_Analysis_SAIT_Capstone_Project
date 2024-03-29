#%%
import pandas as pd
import seaborn as sns

#%% Importing dataset
df_churn = pd.read_csv("autoinsurance_group2.csv")
df_churn.info()

#%% Cleaning the dataset

# Removing customers with income 80,372.176
df_churn = df_churn[df_churn["income"] != 80372.176]

# Removing customers with age > 100, keep only age <= 100
df_churn = df_churn[df_churn["age_in_years"] <= 100]

# Removing cutomers on birthdate 1967-07-07
df_churn = df_churn[df_churn["date_of_birth"] != "1967-07-07"]

#%% Create HOME_MARKET_VALUE_MID from HOME_MARKET_VALUE
hmv = df_churn["home_market_value"]
hmv.unique()

# Replace "1000000 Plus" and drop NaN
hmv = hmv.replace("1000000 Plus", "1000000 - 1000000")
hmv = hmv.dropna()
hmv.info()

# Splitting HOME_MARKET_VALUE
df_hmv = hmv.str.split(" - ", expand=True)
df_hmv[0] = df_hmv[0].astype(int)

# Rounding to integers
df_hmv[1] = df_hmv[1].astype(int) + 1
df_hmv[2] = df_hmv.mean(axis=1).astype(int)
df_hmv # {0: min, 1: max, 2: average of min and max}

#%% Inserting home_market_values back to df_churn
df_churn.insert(12,"home_market_value_min", df_hmv[0])
df_churn.insert(13,"home_market_value_max", df_hmv[1])
df_churn.insert(14,"home_market_value_mid", df_hmv[2])
df_churn.info()

# Alternatively
# df_hmv = df_hmv.rename(columns=
#     {
#         0: "home_market_value_min",
#         1: "home_market_value_max",
#         2: "home_market_value_mid",
#     }
# )
# df_churn = df_churn.join(df_hmv, how= "left", validate="one_to_one",sort=True)
# df_churn.info()

#%% Optional
# Filling null in home_market_value_mid using its mean
avg_hmv = df_churn["home_market_value_mid"].mean()
df_churn["home_market_value_mid"].fillna(avg_hmv, inplace=True)

#%% Encoding martial_status into 0 and 1. 
# We could use Encoder, but since marital_status has only two unique values...
df_churn["marital_status"].replace(["Married", "Single"], [1, 0], inplace=True)

#%% Reordering columns and reset indices

# original_order = df_churn.columns.tolist()
desired_order = [
    'individual_id',
    'address_id',
    'curr_ann_amt',
    'days_tenure',
    'cust_orig_date',
    'age_in_years',
    'date_of_birth',
    'latitude',
    'longitude',
    'city',
    'state',
    'county',
    'home_market_value',
    'home_market_value_min',
    'home_market_value_max',
    'home_market_value_mid',
    'income',
    'has_children',
    'length_of_residence',
    'marital_status',
    'home_owner',
    'college_degree',
    'good_credit',
    'acct_suspd_date',
    'Churn',
]
df_churn = df_churn[desired_order]
df_churn = df_churn.reset_index(drop=True)
df_churn.info()

#%% Saving the cleaned dataset
# df_churn.to_csv("autoinsurance_cleaned_group2.csv")
