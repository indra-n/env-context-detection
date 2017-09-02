import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV


print("Started")
load_string = "../saves/cv_svm_rad_model_aws_repkfold.pickle"
with open(load_string, 'rb') as f:
    cv_model = pickle.load(f)

print(cv_model.best_params_)
mean_result = cv_model.cv_results_['mean_test_score']
mean_std = cv_model.cv_results_['std_test_score']
print(mean_result)
print(mean_std)

save_string = "../saves/cv_svm_rad_repkfold_means.pickle"
with open(save_string, 'wb') as f:
    pickle.dump(mean_result, f)

save_string = "../saves/cv_svm_rad_repkfold_std.pickle"
with open(save_string, 'wb') as f:
    pickle.dump(mean_std, f)


print("Finished")


