# %%
import pandas as pd
import numpy as np
import seaborn as sns
import pickle 
import matplotlib.pyplot as plt
from keras import Sequential, optimizers, initializers
from keras.losses import BinaryFocalCrossentropy
from keras.layers import Dense, Dropout, LeakyReLU
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler #, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, \
    classification_report, confusion_matrix, roc_curve, \
    BinaryAccuracy, BinaryCrossentropy, MeanSquaredError, \
    Precision, Recall, AUC, \
    TruePositives, FalsePositives, TrueNegatives, FalseNegatives

# import fastai
# from fastai.tabular.all import *

dir_folder = "D:/Users/Weidong/Downloads/Auto Insurance churn/"

# %% Loading the cleaned Dataset
df_churn = pd.read_csv(f"{dir_folder}autoinsurance_churn_cleaned.csv")

# Removing customers with age > 65
df_churn = df_churn[df_churn["age_in_years"] <= 65]

# %% Preparing dataset and check for outliars

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

print(f"Number of features: {len(features)}")

# %% Dividing dataset into training and testing sets 80-20
# from sklearn.model_selection import train_test_split

x_train_pre_churn, x_test_pre_churn, y_train_churn, y_test_churn = train_test_split(
    df_churn[features], df_churn["Churn"], train_size=0.8, random_state=123, 
)

# %% Scaling or normalizing dataset

mms = MinMaxScaler()
sc = StandardScaler()

col_mms = [
    "curr_ann_amt", 
    # "days_tenure", 
    "age_in_years", 
    "home_market_value_mid", 
    "income", 
    "has_children", 
    "length_of_residence", 
]

ct = ColumnTransformer(
    [
        ("mms", mms, col_mms)
        # ("sc_scale", sc, features)
    ],
    remainder="passthrough",
)
# %% Normalize the data using MinMaxScaler()

x_train_fnn = ct.fit_transform(x_train_pre_churn)#[:,3:]
x_test_fnn = ct.transform(x_test_pre_churn)#[:,3:]
y_train_fnn = y_train_churn
y_test_fnn = y_test_churn


# %% FNN analysis ---------------------------------------------------------

# Setting initial bias
initial_bias = y_train_fnn.mean()
ln_bias = np.log(initial_bias) # taking ln (log of base e) 
output_bias = initializers.Constant(ln_bias)

# Metrics
METRICS = [
    BinaryAccuracy(name='accuracy'),
    BinaryCrossentropy(name='cross entropy'),  # same as model's loss
    MeanSquaredError(name='Brier score'),
    # TruePositives(name='tp'),
    # FalsePositives(name='fp'),
    # TrueNegatives(name='tn'),
    # FalseNegatives(name='fn'), 
    Precision(name='precision'),
    Recall(name='recall'),
    # AUC(name='auc'),
    # AUC(name='prc', curve='PR'), # precision-recall curve
]

# Building the model

fnn = Sequential()

fnn.add(Dense(units=24, input_shape=(10,), activation=LeakyReLU(0.05)))

fnn.add(Dense(units=12, activation=LeakyReLU(0.05)))

fnn.add(Dense(units=8, activation=LeakyReLU(0.05)))

fnn.add(Dropout(0.5))

fnn.add(Dense(units=4, activation=LeakyReLU(0.05)))

fnn.add(Dense(units=1, activation='sigmoid', bias_initializer=output_bias))

# %% counting the pos and neg labels
neg, pos = np.bincount(y_train_fnn)
total = neg + pos

print(f"Total: {total}\nPositive: {pos} ({100 * pos / total:.2f}% of total)")

# %% Compiling the model
from keras.losses import BinaryFocalCrossentropy

# focused cross entropy loss
bf_loss = BinaryFocalCrossentropy(
    apply_class_balancing=True,
    alpha=0.86,
    gamma=0.2,
)
# simple cross entropy loss
binary_loss = "binary_crossentropy"

opt = optimizers.Adam(learning_rate=0.00001)
fnn.compile(loss=bf_loss, optimizer="adam", metrics=METRICS)

# %% Setting class_weights

# weight_for_0 = (1 / neg) * (total / 2.0)
# weight_for_1 = (1 / pos) * (total / 2.0)
# class_weight = {0: weight_for_0, 1: weight_for_1}

# print('Weight for class 0: {:.2f}'.format(weight_for_0))
# print('Weight for class 1: {:.2f}'.format(weight_for_1))

# %% Training the model
fit_fnn = fnn.fit(x_train_fnn, y_train_fnn, epochs=250, batch_size=4096, )

# %% saving the model and reading the model

# with open("churn_fnn_0.85_0.2_ep250_age65_0.395.pkl", "wb") as file:
#     saved_fnn = pickle.dump(fnn, file)

# with open("churn_fnn_0.85_ep200.pkl", 'rb') as file:  
#     fnn2 = pickle.load(file)

# %% Predictions of testing set using fnn
y_train_pred_fnn = fnn.predict(x_train_fnn)
y_pred_fnn = fnn.predict(x_test_fnn)

# Plotting the preditions
df_y_pred = pd.DataFrame(y_pred_fnn)
sns.displot(df_y_pred)

# %% Calculate the correct percentile of class 1 in y_true. 
# percent of class 1 in y_test_fnn
perc_1 = y_test_fnn.sum()/len(y_test_fnn)
percentile_1 = np.percentile(y_pred_fnn, (1 - perc_1) *100)
percentile_1

# %% Plotting confusion matrix ----------------------------------
def plot_confusion(y_true, y_pred, perc=0.5):
    cl = []
    for pred in y_pred:
        if pred >= perc:
            cl.append(1)
        else:
            cl.append(0)
    print(classification_report(y_true, cl))
    print(f"Threshold Used: {perc}")
    # plotting the confusion matrix
    cm = confusion_matrix(y_true, cl)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# %% Confusion for usual 0.5 cut off
plot_confusion(y_test_fnn, y_pred_fnn, perc=0.5)

# %% Confusion for percentile cut off
plot_confusion(y_test_fnn, y_pred_fnn, percentile_1)

# %% Confusion for manual cut off
plot_confusion(y_test_fnn, y_pred_fnn, 0.395)

# %% Plotting loss value
fig = plt.figure(figsize=(16, 9))
plt.plot(fit_fnn.history["loss"], label="loss")
# plt.plot(fit_db.history["accuracy"], label="accuracy")
plt.show()

# %% Plotting threholds
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    # plt.xlim([-0.5,20])
    # plt.ylim([-0.5,20])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

plot_roc("Train Baseline", y_train_fnn, y_train_pred_fnn, color = "blue")
plot_roc("Test Baseline", y_test_fnn, y_pred_fnn, color = "red", linestyle='--')
plt.legend(loc='lower right')

#%% Renaming df to plot correlation matrix
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
    "Churn",
]
df2 = df_churn
df2 = df2[features]
df2.columns = features2

# %% Plotting correlation matrix
correlation_matrix = df2.corr()

# Heatmap for correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=.5,cbar=False)
plt.title('Correlation Matrix')
plt.show()
