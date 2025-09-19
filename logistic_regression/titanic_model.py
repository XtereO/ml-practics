import sys
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def get_fit_model(data, model_index):
    label = "survived"
    X = data.drop(label, axis="columns")
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=data[label])

    model = LogisticRegression(random_state=7, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"model {model_index} has f1: {f1_score(y_test, y_pred)}")
    return model

def map_honorific(h):
    if h in ["Mlle", "Ms"]:
        return "Miss"
    if h in ["Dona", "Countess", "theCountess"]:
        return "Mrs"
    if h in ["Rev", "Col", "Dr", "Major", "Don", "Capt"]:
        return "Mr"
    return h

def apply_age(mean_ages):
    def _apply_age(r):    
        if math.isnan(r["age"]):
            r["age"] = mean_ages[r["honorific"]]
        return r
   
    return _apply_age

data = pd.read_csv("titanic_train.csv")
print("missed age: ", data["age"].isna().sum())
print("survived/total:", data["survived"].sum(), "/", data.shape[0])

third = data.shape[0]/3
for col in data.columns:
    if data[col].isna().sum() > third:
        data.drop(col, axis="columns", inplace=True)
        print("removed col", col)

data["fam_size"] = data["sibsp"] + data["parch"]
data.drop(["sibsp", "parch", "ticket"], axis="columns", inplace=True)
print(data.describe())
print(data.columns, data.shape)
print("female with pclass=1:", data.loc[(data["sex"]=="female") & (data["pclass"]==1), "survived"].sum())
print("survived female with pclass=1:", data.loc[(data["sex"]=="female") & (data["pclass"]==1), "survived"].count())

if sys.argv[1]=="1":
    data_survived = data.loc[data["survived"]==1]
    predictors = list(set(data.columns)-set(["survived"]))
    for col in predictors:
        if pd.api.types.is_numeric_dtype(data_survived[col]): 
            n_groups = int((data_survived[col].max()-data_survived[col].min())**0.5)
            sns.histplot(data_survived[col], bins=n_groups, kde=True)
        else:
            vc = data_survived[col].value_counts()
            sns.barplot(x=vc.index, y=vc.values)
        plt.show()

data_numeric = data.select_dtypes(exclude="object")
print("data_numeric\n", data_numeric.describe(), data_numeric.columns)
data_numeric_filtrated = data_numeric.dropna()
print("data_numeric_filtrated\n", data_numeric_filtrated.describe())
get_fit_model(data_numeric_filtrated, 1)

data_numeric_mean = data_numeric.copy()
data_numeric_mean["age"].fillna(data_numeric_mean["age"].mean(), inplace=True)
print("data_numeric_mean\n", data_numeric_mean.describe())
get_fit_model(data_numeric_mean, 2)

data_name = data_numeric.loc[:].join(data.loc[:, "name"])
data_name["honorific"] = data_name["name"].map(lambda n: n.split(",")[1].split(".")[0].replace(" ", ""))
print(data_name["honorific"].unique())
data_name["honorific"] = data_name["honorific"].map(map_honorific)
print(data_name["honorific"].unique())
print("number Master and Mr: ", data_name[data_name["honorific"]=="Master"].count(), data_name[data_name["honorific"]=="Mr"].count())
mean_ages = {"Miss": data_name[data_name["honorific"]=="Miss"]["age"].mean(), 
             "Mr": data_name[data_name["honorific"]=="Mr"]["age"].mean(),
             "Master": data_name[data_name["honorific"]=="Master"]["age"].mean(),
             "Mrs": data_name[data_name["honorific"]=="Mrs"]["age"].mean()}
print("mean age of Miss", data_name[data_name["honorific"]=="Miss"]["age"].mean())

data_name = data_name.apply(apply_age(mean_ages), axis="columns")
print(data_name.describe())
get_fit_model(data_name.drop(["name", "honorific"], axis="columns"), 3)

data_encoded = data_name.drop(["name", "honorific"], axis="columns")
print("data_encoded structure",data_encoded.columns, data_encoded.shape)
data_encoded["sex"] = data["sex"]
data_encoded["embarked"] = data["embarked"]

data_encoded = pd.get_dummies(data_encoded, drop_first=True, columns=["sex", "embarked"])
get_fit_model(data_encoded, 3)
