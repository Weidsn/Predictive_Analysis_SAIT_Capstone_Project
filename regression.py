#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

#%% Step 1: Dividing dataset into training and testing sets 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% Step 2: Normalizing features
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

#%% Step 3: Applying SMOTE to the training set
smote = SMOTE(random_state=123)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

#%% Step 4: Training Logistic Regression model on the resampled data
logistic_regression_model = LogisticRegression(random_state=123)
logistic_regression_model.fit(X_train_resampled, y_train_resampled)

#%% Step 5: Making predictions on the testing set
y_pred = logistic_regression_model.predict(X_test_scaled)

#%% Step 6: Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
plot_confusion(y_test, y_pred)

#%% Visualization
# Bar plot for categorical variables after SMOTE
categorical_vars = ['income', 'good_credit', 'has_children', 'marital_status', 'college_degree', 'home_owner']
for var in categorical_vars:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=var, hue='Churn', data=df)
    plt.title(f'Distribution of {var} by Churn')
    plt.show()

#%% Box plot for numerical variables after SMOTE
numerical_vars = ['curr_ann_amt', 'length_of_residence', 'home_market_value_mid', 'age_in_years']
for var in numerical_vars:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y=var, data=df)
    plt.title(f'{var} by Churn')
    plt.show()

#%% Plotting correlation matrix
# Remaning columns 
features1 = [
    'curr_ann_amt',
    'age_in_years',
    'home_market_value_mid',
    'income',
    'has_children',
    'length_of_residence',
    'marital_status',
    'home_owner',
    'college_degree',
    'good_credit',
    "Churn"
]

features2 = [
    'Annual Premium',
    'Age',
    'Home Market Value',
    'Income',
    'Has Children',
    'Lengh of Residence',
    'Marital Status',
    'Home Owner',
    'College Degree',
    'Good Credit',
    "Churn"
]

df2 = df[features1]
df2.columns = features2

#%% Plotting the matrix
correlation_matrix = df2.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5,cbar=True,)
plt.title('Correlation Matrix')
plt.show()

#%% 
