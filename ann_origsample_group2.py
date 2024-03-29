#%%
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import helper_group2 # Import custom helper functions for model evaluation

#%% Importing dataset
df = pd.read_csv("autoinsurance_cleaned_group2.csv")
df = df[df["age_in_years"] <= 65]

#%%
features = [
    "curr_ann_amt", 
    # "days_tenure", 
    "age_in_years", 
    "home_market_value_mid", 
    "income", 
    "has_children", 
    "length_of_residence", 
    "marital_status",
    "home_owner", 
    "college_degree", 
    "good_credit",
]

X = df[features]
y = df['Churn']

#%% Step 1: Dividing dataset into training and testing sets 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% Setp 2: Normalizing features
mms = MinMaxScaler()

col_mms = [
    "curr_ann_amt", 
    "age_in_years", 
    "home_market_value_mid", 
    "income", 
    "has_children", 
    "length_of_residence", 
]

ct = ColumnTransformer(
    [("mms", mms, col_mms)],
    remainder="passthrough",
)

X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

#%% Step 3: Loading the model
with open("churn_fnn_0.85_0.2_ep250_age65_0.395.pkl", 'rb') as file:  
    fnn = pickle.load(file)

# Step 4: Making predictions on the testing set
y_train_pred = fnn.predict(X_train_scaled)
y_pred = fnn.predict(X_test_scaled)

#%% Step 5: Evaluating the model

# Plotting the preditions
sns.displot(pd.DataFrame(y_pred))

# Calculating the correct percentile of class 1 in y_true. 
perc_1 = y_test.sum()/len(y_test) # Percent of class 1 in y_test
percentile_1 = np.percentile(y_pred, (1 - perc_1) *100)
print(f"Percentile threshold: {percentile_1}.")

# Plotting confusion matrix for manual threshold
helper_group2.plot_confusion(y_test, y_pred, 0.395)

# Plotting roc curves for the training and testing sets
helper_group2.plot_roc("Train Baseline", y_train, y_train_pred, color = "blue")
helper_group2.plot_roc("Test Baseline", y_test, y_pred, color = "red", linestyle='--')
plt.legend(loc='lower right')

#%%
