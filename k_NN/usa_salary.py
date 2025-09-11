import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

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
