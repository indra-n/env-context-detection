import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.preprocessing import MinMaxScaler

cols = ['num_sat_25', 'sum_snr_25',
        'num_sat', 'sum_snr',
        'elev_0_30', 'elev_30_60', 'elev_60_90',
        'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
        'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
        'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25']

# My training data
with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_df = pickle.load(f)
# # Load level 1 training data
# with open('..\saves\meta_train_data_2.pickle', 'rb') as f:
#     meta_train_data = pickle.load(f)
#
# with open('..\saves\meta_train_labels_2.pickle', 'rb') as f:
#     meta_train_true_labels = pickle.load(f)


# My test data
with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    test_df = pickle.load(f)



train_data = train_df
test_data = test_df

#########
# Experiment with Han's data
# with open('..\saves\given_data_sample.pickle', 'rb') as f:
#     given_data = pickle.load(f)
#
# inter_train_df = [train_data, test_data]
# inter_train = pd.concat(inter_train_df).sample(frac=1).reset_index(drop=True)
#
# train_data = inter_train
# test_data = given_data

#######


print(train_data['true_class'].value_counts())

scaler = MinMaxScaler()
scaler.partial_fit(train_data[cols])
scaler.partial_fit(test_data[cols])

train_input = scaler.transform(train_data[cols])
test_input = scaler.transform(test_data[cols])


forest = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=57) # , learning_rate=1.0, max_depth=1, random_state=0)
forest.fit(train_input, train_data['true_class'])

pred = forest.predict(test_input)
pred_probas = forest.predict_proba(test_input)

differ = abs(pred - test_data['true_class'])
accu = 1 - np.count_nonzero(differ) / test_data.shape[0]

print(accu)

cm = confusion_matrix(test_data['true_class'], pred)

print(cm)
cm_proc = cm / np.sum(cm, axis=1).reshape((4, 1))
np.set_printoptions(suppress=True)
print(cm_proc)


wrong_= test_data[differ != 0]
wrong_pred = wrong_.loc[wrong_['true_class'] == 1]
print(wrong_.shape)
print(wrong_['location_id'].value_counts())
