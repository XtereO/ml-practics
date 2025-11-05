from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.model_selection import train_test_split 
from keras.datasets import mnist
import warnings
warnings.filterwarnings("ignore")

(_,_), (X, y) = mnist.load_data()
print("data shape", X.shape, y.shape)

X = X.reshape(X.shape[0], -1)
print("flat data shape", X.shape)

rs=39
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
model = KMeans(n_clusters=10, random_state=rs).fit(X_train)
m=7
labels = model.labels_
cluster_sizes = np.bincount(labels)
print(cluster_sizes)

y_tr_pred = model.predict(X_train)
y_pred = model.predict(X)
print(y_tr_pred[y_tr_pred==m].shape, y_pred[y_pred==m].shape)
