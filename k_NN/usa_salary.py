import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import seaborn as sns
import matplotlib.pyplot as plt

def get_fit_model(X_train, X_test, y_train, y_test, index):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"f1_score of model {index}: {f1_score(y_test, y_pred)}")

    return model

data = pd.read_csv("usa_salary_data_train.csv")
data_numeric = data.select_dtypes(exclude=['object'])
print(data.columns, data.describe())
print(data[data.label==0].shape, data_numeric.shape)

X = data_numeric[["age", "fnlwgt", "education-num", "capital-gain" , "capital-loss", "hours-per-week"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=data["label"])
print("description X_train:", X_train.describe())

get_fit_model(X_train, X_test, y_train, y_test, 1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("means of X_train_scaled:", X_train_scaled[:, 1].mean())

get_fit_model(X_train_scaled, X_test_scaled, y_train, y_test, 2)


data_excl_ed_mart = data.drop(["education", "marital-status"], axis=1)

str_columns = data_excl_ed_mart.select_dtypes(include=['object']).columns
if sys.argv[1]=="1":
    for feature in str_columns:
        sns.barplot(data_excl_ed_mart[feature], estimator="sum")
        plt.show()

data_excl_ed_mart.replace("?", np.nan, inplace=True)
print("rows with at least one missed value:", data_excl_ed_mart.isna().any(axis=1).sum())
data_filtered = data_excl_ed_mart.dropna()
print("shape of filtered data:", data_filtered.shape)
data_encoded = pd.get_dummies(data_filtered, drop_first=True)
print("shape of encoded data", data_encoded)

X_enc = data_encoded.drop("label", axis=1)
y_enc = data_encoded["label"]
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_enc, y_enc, test_size=0.2, random_state=21, stratify=data_encoded["label"])
scaler = MinMaxScaler()
X_train_e_scaled = scaler.fit_transform(X_train_e)
X_test_e_scaled = scaler.transform(X_test_e)

get_fit_model(X_train_e_scaled, X_test_e_scaled, y_train_e, y_test_e, 3)

data_full = data_excl_ed_mart
'''for feature in str_columns:
    col = data_full[feature]
    col_freq = col.value_counts()
    most_freq_value = col_freq.index[0]
    data_full[feature].replace(np.nan, most_freq_value, inplace=True)
    print(feature, col_freq.index[0])'''
for feature in str_columns:
    most_freq_value = data_full[feature].mode()[0]
    data_full[feature].fillna(most_freq_value, inplace=True)

data_full_encoded = pd.get_dummies(data_full, drop_first=True)
X_full = data_full_encoded.drop("label", axis=1)
y_full = data_full_encoded["label"]
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y_full, test_size=0.2, random_state=21, stratify=data_full_encoded["label"])
scaler = MinMaxScaler()
X_train_f_scaled = scaler.fit_transform(X_train_f)
X_test_f_scaled = scaler.transform(X_test_f)

get_fit_model(X_train_f_scaled, X_test_f_scaled, y_train_f, y_test_f, 4)
