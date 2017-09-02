import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.preprocessing import MinMaxScaler


with open('../saves/meta_train_data.pickle', 'rb') as f:
    train_data = pickle.load(f)

with open('../saves/meta_train_labels.pickle', 'rb') as f:
    train_true_labels = pickle.load(f)

with open('../saves/meta_train_groups.pickle', 'rb') as f:
    train_groups = pickle.load(f)

param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']
}

print("Started")
cv_estimator = LeaveOneGroupOut()
CV_rfc = GridSearchCV(LogisticRegression(random_state=57), param_grid=param_grid, cv=cv_estimator)
CV_rfc.fit(train_data, train_true_labels, train_groups)

print(CV_rfc.best_params_)
print(CV_rfc.cv_results_['mean_test_score'])

with open('../saves/cv_meta_LG_model.pickle', 'wb') as f:
    pickle.dump(CV_rfc, f)

print("Finished")
