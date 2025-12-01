import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def sys_arg(value, index=1):
    return len(sys.argv) > index and sys.argv[index] == value


def cr(title, y_true, y_pred):
    print(f"{title}\n {classification_report(y_true, y_pred, digits=4)}")


data = pd.read_csv("electricity_train.csv")
print("general info", data.describe())
print("corr matrix\n", data.corr())
print("value counts of classes", data["class"].value_counts())

if sys_arg("data_hist"):
    for feature in data.columns:
        plt.title(feature)
        plt.hist(data[feature])
        plt.show()

X = data.drop("class", axis="columns")
y = data["class"]
RS = 13
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RS)

if sys_arg("rf"):
    rf = RandomForestClassifier(random_state=RS).fit(X_train, y_train)
    print(classification_report(y_train, rf.predict(X_train)))

    '''
    params_grid = {'n_estimators': [100, 300, 500],
                   'max_leaf_nodes': list(range(6, 10)),
                   'min_samples_leaf': [1, 2, 3]}
    
    searching = GridSearchCV(RandomForestClassifier(
        bootstrap=False,
        class_weight='balanced',
        n_jobs=-1,
        max_features='sqrt',
        random_state=RS),
        params_grid,
        verbose=4,
        cv=3).fit(X_train, y_train)
    print("the best model for RF",
          searching.best_estimator_, searching.best_params_)
    '''
    # these params were obtained by "searching"
    rf = RandomForestClassifier(random_state=RS, max_leaf_nodes=9,
                                min_samples_leaf=1, n_estimators=100, bootstrap=False, class_weight='balanced', n_jobs=-1).fit(X_train, y_train)
    cr("rf train", y_train, rf.predict(X_train))
    cr("rf test", y_test, rf.predict(X_test))
    print("the most important features: ", rf.feature_importances_)

if sys_arg("vc"):
    vc = VotingClassifier(estimators=[("lr", LogisticRegression(solver='liblinear', random_state=RS)), ("svc", SVC(
        random_state=RS)), ("sgd", SGDClassifier(random_state=RS))], voting="hard").fit(X_train, y_train)
    cr("vc test", y_test, vc.predict(X_test))

if sys_arg("bg"):
    bg = BaggingClassifier(
        DecisionTreeClassifier(class_weight='balanced'),
        max_samples=0.5,
        max_features=0.5,
        bootstrap=False,
        random_state=RS).fit(X_train, y_train)
    cr("bg test", y_test, bg.predict(X_test))

if sys_arg("gb"):
    gb = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.8, random_state=RS, max_depth=2).fit(X_train, y_train)
    cr("gb test", y_test, gb.predict(X_test))

if sys_arg("ab"):
    ab = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
                            n_estimators=300,
                            learning_rate=0.5,
                            random_state=RS).fit(X_train, y_train)
    cr("ab test", y_test, ab.predict(X_test))

if sys_arg("st"):
    st = StackingClassifier(estimators=[("rf", RandomForestClassifier(random_state=RS)), ("svc", SVC(
        random_state=RS))], final_estimator=LogisticRegression(random_state=RS)).fit(X_train, y_train)
    cr("st test", y_test, st.predict(X_test))

if sys_arg("pred"):
    X, y = X_train, y_train
    data_reserved = pd.read_csv("electricity_reserved.csv")
    pca = PCA(svd_solver="full", n_components=0.99999).fit(X)
    mm = MinMaxScaler()
    print("explained ratio", pca.n_components_)
    X_reduced = (pca.transform(X))
    X_pred = (pca.transform(data_reserved))
    X_reduced = mm.fit_transform(X_reduced)
    X_pred = mm.transform(X_pred)

    '''
    params_grid = {'n_estimators': [500, 700, 900],
                   'learning_rate': [0.7],
                   'max_depth': [4]}
    
    searching = GridSearchCV(GradientBoostingClassifier(),
        params_grid,
        verbose=4,
        cv=3).fit(X, y)

    print("the best model for RF",
          searching.best_estimator_, searching.best_params_)
    '''
    # it turned out that the most impact param for GB is max_depth (more max_depth the more score I usually get in GridSearch)
    # learning_rate influences on speed of learning, n_estimators can influence on score a bit too
    model = GradientBoostingClassifier(
        n_estimators=900, learning_rate=0.7, max_depth=4)

    cr("pred train", y, model.predict(X))
    y_pred = (model.predict(data_reserved))
    pd.DataFrame([y_pred]).to_csv('predictions.csv', index=False, header=False)
