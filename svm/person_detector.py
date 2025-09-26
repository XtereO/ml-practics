import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

data = pd.read_csv("persons_pics_train.csv")
print("data size:", data.shape)

unique_persons= data["label"].unique()
print("unique persons:", unique_persons, unique_persons.shape)
data_persons = data.groupby("label")
freq_data = data_persons.size()
print("freq person data:", freq_data)
'''sns.barplot(data=freq_data)
plt.show()'''

mean_data = data_persons.mean()
print("mean data:", mean_data)
print(mean_data.loc["Gerhard Schroeder", "0"])

'''
for person in unique_persons:
    plt.imshow(mean_data.loc[person].astype(float).to_numpy().reshape(62,47), cmap='gray')
    plt.axis('off')
    plt.title(person)
    plt.show()
'''

gs = mean_data.loc["Gerhard Schroeder"]
hc = mean_data.loc["Hugo Chavez"]
cos_similarity = (gs @ hc) / (np.linalg.norm(gs)*np.linalg.norm(hc))
print("cos similarity of mean Gerhard Schroeder and Hugo Chavez", cos_similarity)

X = data.drop("label", axis=1)
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=data["label"])

model1= SVC(kernel="linear", random_state=7)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(f'f1 weighted score of model1 is {f1_score(y_pred=y_pred, y_true=y_test, average="weighted")}')

if sys.argv[1] == "1":
    tuned_parameters = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000], 'class_weight': [None, 'balanced'], 'random_state':[7]}]
    cv = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=3)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)

    print("the best params of model is", cv.best_params_)
    print("the best f1 score of the best model is", f1_score(y_pred=y_pred, y_true=y_test, average="weighted"))

reducing = PCA(svd_solver="full")
reducing.fit(X_train)

explained_variance = 0
for (i, v) in enumerate(reducing.explained_variance_ratio_):
    explained_variance+=v
    if explained_variance > 0.95:
        print(f'{i+1} components are enough to explain {explained_variance}>0.95 data')
        break
reducing = PCA(n_components=i+1).fit(X_train)

X_train_reduced = reducing.transform(X_train)
X_test_reduced = reducing.transform(X_test)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4],
                     'C': [1000], 'class_weight': [None, 'balanced'], 'random_state':[7]}]
cv = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=3)
cv.fit(X_train_reduced, y_train)
y_pred = cv.predict(X_test_reduced)
print(f'f1 weighted score of model2 is {f1_score(y_pred=y_pred, y_true=y_test, average="weighted")}')
