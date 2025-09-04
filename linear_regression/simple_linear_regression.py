from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

data = pd.read_csv("simple_lin_regression_data.csv")
X, Y = data[["X"]], data["Y"]
print(f"mean X, Y: {X.mean()}, {Y.mean()}")

model = LinearRegression()
model.fit(X, Y)
print(f"y = {model.coef_}x + {model.intercept_}")
print(f"R^2 score: {r2_score(Y, model.predict(X))}")