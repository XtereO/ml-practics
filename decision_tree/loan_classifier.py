import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


def prepare_data(X_train, X_test, y_train, y_test):
    def get_fit_model(index, criterion, class_weight=None):
        model = DecisionTreeClassifier(
            random_state=121, criterion=criterion, class_weight=class_weight)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"f1 score of model {index}", f1_score(y_test, y_pred))
        return model
    return get_fit_model


data = pd.read_csv("Bank_Personal_Loan_Modelling_train.csv")
data = data.drop("ID", axis=1)
print("data head", data.describe())

if len(sys.argv) > 1 and sys.argv[1] == "1":
    sns.heatmap(data=data.corr(), annot=True)
    plt.show()

data = data.drop("ZIP Code", axis=1)

label = "Personal Loan"
data["Experience"] -= data["Experience"].min()
print("mean shifted exp", data["Experience"].mean())

data["CCAvg"] *= 12
print("mean multiplied CCAvg", data["CCAvg"].mean())

cols = data.columns
if len(sys.argv) > 1 and sys.argv[1] == "2":
    for col in cols:
        plt.title(col)
        sns.boxplot(data[col])
        plt.show()

if len(sys.argv) > 1 and sys.argv[1] == "3":
    sns.histplot(data=data["Mortgage"])
    plt.show()

z_mortgage = zscore(data["Mortgage"])
print("Mortgage |zscore|>3", z_mortgage[abs(z_mortgage) > 3].size)

data = data[abs(z_mortgage) <= 3]
print("Remain data size", data.shape)

X = data.drop("Personal Loan", axis=1)
y = data["Personal Loan"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=121, stratify=data["Personal Loan"])

get_fit_model1 = prepare_data(X_train, X_test, y_train, y_test)
model1 = get_fit_model1(1, "gini")
model2 = get_fit_model1(2, "entropy")
model3 = get_fit_model1(3, "gini", "balanced")
model4 = get_fit_model1(4, "entropy", "balanced")

if len(sys.argv) > 1 and sys.argv[1] == "4":
    sns.histplot(y_train)
    plt.show()

print("number of accepted Personal Loan",
      y_train[y_train == 1].size, y_train.size)

sm = SMOTE(random_state=121)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
print("number of accepted Personal Loan after resampling",
      y_resampled[y_resampled == 1].size, y_resampled.size)

get_fit_model2 = prepare_data(X_resampled, X_test, y_resampled, y_test)
model5 = get_fit_model2(5, "gini")
model6 = get_fit_model2(6, "entropy")
