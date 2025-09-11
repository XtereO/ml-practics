import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("k_NN_base.csv")
X = data[["X", "Y"]]
y = data["Class"]

point_X = pd.DataFrame({"X": [34], "Y": [28]})
print(point_X.loc[:])

model2 = KNeighborsClassifier(n_neighbors=3, p=2)
model2.fit(X, y)

closet_neigh2_point = model2.kneighbors(point_X, n_neighbors=3, return_distance=True)
pred_class2_point = model2.predict(point_X)
print("Euclid metric:", closet_neigh2_point, pred_class2_point)

model1 = KNeighborsClassifier(n_neighbors=3, p=1)
model1.fit(X, y)

closet_neigh1_point = model1.kneighbors(point_X, n_neighbors=3, return_distance=True)
pred_class1_point = model1.predict(point_X)
print("Manhattan distance:", closet_neigh1_point, pred_class1_point)
