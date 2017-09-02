import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
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
    'n_neighbors': [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                    150, 160, 170, 180, 190, 200, 210, 220, 230]
}

print("Started")
cv_estimator = LeaveOneGroupOut()

CV_rfc = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv_estimator)
CV_rfc.fit(train_data, train_true_labels, train_groups)

print(CV_rfc.best_params_)

with open('..\saves\cv_meta_knn_model.pickle', 'wb') as f:
    pickle.dump(CV_rfc, f)

print("Finished")
