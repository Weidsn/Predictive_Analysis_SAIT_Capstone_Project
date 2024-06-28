#%% importing dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from helper_model_evaluation import *

#%% importing dataset
df = pd.read_csv("autoinsurance_cleaned_group2.csv")
df = df[df["age_in_years"] <= 65]

#%%
features = [
    'curr_ann_amt',
    # "days_tenure", 
    'age_in_years',
    'home_market_value_mid',
    'income',
    'has_children',
    'length_of_residence',
    'marital_status',
    'home_owner',
    'college_degree',
    'good_credit',
]

X = df[features]
y = df['Churn']

#%% Step 1: Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% Step 2: Applying SMOTE to the training set
smote = SMOTE(random_state=123)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#%% Step 3: Training the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=123)
dt_classifier.fit(X_train_resampled, y_train_resampled)

#%% Step 4: Making predictions on the test set
y_pred = dt_classifier.predict(X_test)

#%% Step 5: Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
plot_confusion(y_test, y_pred)

# %%
