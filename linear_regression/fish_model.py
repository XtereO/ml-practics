from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("fish_train.csv")
fish_features = ["Species", "Length1", "Length2", "Length3", "Height", "Width"]
X, y = data[fish_features], data["Weight"]

train_X, val_X, train_Y, val_Y = train_test_split(X, y, stratify=data["Species"], test_size=0.2, random_state=41)
train_X = train_X.drop("Species", axis=1)
val_X = val_X.drop("Species", axis=1)
print(f"train_X[Width] mean: {train_X["Width"].mean()}")

lin_model_1 = LinearRegression()
lin_model_1.fit(train_X, train_Y)
pred_Y_1 = lin_model_1.predict(val_X)
print(f"R^2 of lin model 1: {r2_score(val_Y, pred_Y_1)}")

'''sns.heatmap(train_X)
plt.show()
print(train_X.corr())'''
fish_correlated_features = ["Length1", "Length2", "Length3"]
model_pca = PCA(svd_solver="full", n_components=1)
model_pca.fit(train_X[fish_correlated_features])
print(model_pca.components_, model_pca.explained_variance_ratio_)
lengths = (train_X[fish_correlated_features]-train_X[fish_correlated_features].mean()) @ model_pca.components_[0]

train_reduced_X = train_X.drop(fish_correlated_features, axis=1)
train_reduced_X["Lengths"] = lengths
val_reduced_X = val_X.drop(fish_correlated_features, axis=1)
val_reduced_X["Lengths"] = model_pca.transform(val_X[fish_correlated_features])

lin_model_2 = LinearRegression()
lin_model_2.fit(train_reduced_X, train_Y)
pred_Y_2 =  lin_model_2.predict(val_reduced_X)
print(f"R^2 of lin model 2: {r2_score(val_Y, pred_Y_2)}")

train_red_cube_X = train_reduced_X ** 3
val_red_cube_X = val_reduced_X ** 3
print(f"train_red_cube_X[Width] mean: {train_red_cube_X.mean()}")
lin_model_3 = LinearRegression()
lin_model_3.fit(train_red_cube_X, train_Y)
pred_Y_3 = lin_model_3.predict(val_red_cube_X)
print(f"R^2 of lin model 3: {r2_score(val_Y, pred_Y_3)}")
