from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_fit_model(train_X, val_X, train_Y, val_Y, index):
    model = LinearRegression()
    model.fit(train_X, train_Y)
    pred_Y = model.predict(val_X)
    print(f"R^2 of lin model {index}: {r2_score(val_Y, pred_Y)}")

    return model

data = pd.read_csv("fish_train.csv")
fish_features = ["Species", "Length1", "Length2", "Length3", "Height", "Width"]
X, y = data[fish_features], data["Weight"]

_train_X, _val_X, train_Y, val_Y = train_test_split(X, y, stratify=data["Species"], test_size=0.2, random_state=41)
train_X = _train_X.drop("Species", axis=1)
val_X = _val_X.drop("Species", axis=1)
print(f"train_X[Width] mean: {train_X["Width"].mean()}")

get_fit_model(train_X, val_X, train_Y, val_Y, 1)


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

get_fit_model(train_reduced_X, val_reduced_X, train_Y, val_Y, 2)


train_red_cube_X = train_reduced_X ** 3
val_red_cube_X = val_reduced_X ** 3
print(f"train_red_cube_X[Width] mean: {train_red_cube_X.mean()}")

get_fit_model(train_red_cube_X, val_red_cube_X, train_Y, val_Y, 3)


species_dummies_train = pd.get_dummies(_train_X, columns=["Species"])
species_dummies_val = pd.get_dummies(_val_X, columns=["Species"])
train_red_cube_s_X = train_red_cube_X.copy(deep=False)
val_red_cube_s_X = val_red_cube_X.copy(deep=False)
all_species = [item for item in species_dummies_train.columns if item not in [*fish_correlated_features, *train_red_cube_X.columns] ]
print(all_species) 
for species in all_species:
    train_red_cube_s_X[species] = species_dummies_train[species]
    val_red_cube_s_X[species] = species_dummies_val[species]

get_fit_model(train_red_cube_s_X, val_red_cube_s_X, train_Y, val_Y, 4)

species_dummies_d_train = pd.get_dummies(_train_X, columns=["Species"], drop_first=True)
species_dummies_d_val = pd.get_dummies(_val_X, columns=["Species"], drop_first=True)
train_red_cube_sd_X = train_red_cube_X.copy(deep=False)
val_red_cube_sd_X = val_red_cube_X.copy(deep=False)
all_d_species = [item for item in species_dummies_d_train.columns if item not in [*fish_correlated_features, *train_red_cube_X.columns] ]
print(all_d_species)
for species in all_d_species:
    train_red_cube_sd_X[species] = species_dummies_d_train[species]
    val_red_cube_sd_X[species] = species_dummies_d_val[species]

get_fit_model(train_red_cube_sd_X, val_red_cube_sd_X, train_Y, val_Y, 5)
