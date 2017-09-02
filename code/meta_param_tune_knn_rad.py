import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.preprocessing import MinMaxScaler


with open('..\saves\meta_train_data.pickle', 'rb') as f:
    train_data = pickle.load(f)

with open('..\saves\meta_train_labels.pickle', 'rb') as f:
    train_true_labels = pickle.load(f)

with open('..\saves\meta_train_groups.pickle', 'rb') as f:
    train_groups = pickle.load(f)

param_grid = {
    'radius': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
               2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.4]
}

print("Started")
cv_estimator = LeaveOneGroupOut()
CV_rfc = GridSearchCV(RadiusNeighborsClassifier(), param_grid=param_grid, cv=cv_estimator)
CV_rfc.fit(train_data, train_true_labels, train_groups)
print(CV_rfc.best_params_)

with open('..\saves\\cv_meta_knn_rad_model.pickle', 'wb') as f:
    pickle.dump(CV_rfc, f)

print("Finished")

