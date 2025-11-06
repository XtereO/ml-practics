import numpy as np
from copy import deepcopy

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from keras.datasets import mnist
import warnings
warnings.filterwarnings("ignore")


def max_mark_digit(mark, labels, y_train):
    max_d = 0, 0
    for d in range(10):
        n_lid = labels[(labels == mark) & (y_train == d)].shape[0]
        if n_lid > max_d[1]:
            max_d = d, n_lid
        # print(f"mark {i} contains digit {d}: {n_lid} times")

    return max_d


(_, _), (X, y) = mnist.load_data()
print("data shape", X.shape, y.shape)

X = X.reshape(X.shape[0], -1)
print("flat data shape", X.shape)

rs = 39
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rs)
model = KMeans(n_clusters=10, random_state=rs).fit(X_train)
m = 7
labels = model.labels_
cluster_sizes = np.bincount(labels)
print(cluster_sizes)

y_tr_pred = model.predict(X_train)
print(y_tr_pred[y_tr_pred == m].shape)

dy_train = deepcopy(y_train)
dy_test = deepcopy(y_test)
pred_labels = model.predict(X_test)

for i in range(10):
    max_d = max_mark_digit(i, labels, y_train)
    dy_train[labels == i] = max_d[0]
    dy_test[pred_labels == i] = max_d[0]
    print(f"the digit of mark {i} is {max_d[0]}")

print(f"[train]: rough accuracy is {accuracy_score(y_train, dy_train)}")
print(f"[test]: rough accuracy is {accuracy_score(y_test, dy_test)}")
print("confusion matrix", confusion_matrix(y_train, dy_train))

X_train_transformed = TSNE(
    n_components=2, init="random", random_state=rs).fit_transform(X_train)
model2 = KMeans(n_clusters=10, random_state=rs).fit(X_train_transformed)
dy_train2 = deepcopy(y_train)
labels2 = model2.labels_

for i in range(10):
    max_d = max_mark_digit(i, labels2, y_train)
    dy_train2[labels2 == i] = max_d[0]
    print(f"the digit of mark {i} is {max_d[0]}")
print(f"[train2]: rough accuracy is {accuracy_score(y_train, dy_train2)}")
