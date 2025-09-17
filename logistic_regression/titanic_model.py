import pandas as pd

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
print("survived female with pclass=1: ", data.loc[(data["sex"]=="female") & (data["pclass"]==1), "survived"].sum())
