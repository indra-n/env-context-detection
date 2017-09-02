import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv
import pickle

#Load the data
fp_indoor_bm = '..\data\indoor\British museum\Obs_features_ext'
fp_indoor_ch2221 = '..\data\indoor\Chadwick 2221\Obs_features_ext'
fp_indoor_ch103a = '..\data\indoor\Chadwick 103 by the window\A\Obs_features_ext'
fp_indoor_ch103b = '..\data\indoor\Chadwick 103 by the window\B\Obs_features_ext'
fp_indoor_jah = '..\data\indoor\JAH363A\Obs_features_ext'

fp_inter = '..\data\intermediate\P1\Obs_features_ext'

fp_urban_p1b = '..\data\\urban\P1B\Obs_features_ext'
fp_urban_p2b = '..\data\\urban\P2B\Obs_features_ext'
fp_urban_p3b = '..\data\\urban\P3B\Obs_features_ext'
fp_urban_p4b = '..\data\\urban\P4B\Obs_features_ext'

fp_open_reg = '..\data\open-sky\P1_REGENT\Obs_features_ext'
fp_open_hyde = '..\data\open-sky\P4_HYDE\Obs_features_ext'

df_indoor_bm = pd.read_csv(fp_indoor_bm)
df_indoor_ch2221 = pd.read_csv(fp_indoor_ch2221)
df_indoor_ch103a = pd.read_csv(fp_indoor_ch103a)
df_indoor_ch103b = pd.read_csv(fp_indoor_ch103b)
df_indoor_jah = pd.read_csv(fp_indoor_jah)
df_indoor_bm['true_class'] = 1
df_indoor_ch2221['true_class'] = 1
df_indoor_ch103a['true_class'] = 1
df_indoor_ch103b['true_class'] = 1
df_indoor_jah['true_class'] = 1

df_inter = pd.read_csv(fp_inter)
df_inter['true_class'] = 2

df_urban_p1b = pd.read_csv(fp_urban_p1b)
df_urban_p2b = pd.read_csv(fp_urban_p2b)
df_urban_p3b = pd.read_csv(fp_urban_p3b)
df_urban_p4b = pd.read_csv(fp_urban_p4b)
df_urban_p1b['true_class'] = 3
df_urban_p2b['true_class'] = 3
df_urban_p3b['true_class'] = 3
df_urban_p4b['true_class'] = 3


df_open_reg = pd.read_csv(fp_open_reg)
df_open_hyde = pd.read_csv(fp_open_hyde)
df_open_reg['true_class'] = 4
df_open_hyde['true_class'] = 4

indoor_dfs = [df_indoor_bm, df_indoor_ch2221, df_indoor_ch103a, df_indoor_ch103b, df_indoor_jah]
urban_dfs = [df_urban_p1b, df_urban_p2b, df_urban_p3b, df_urban_p4b]
open_dfs = [df_open_reg, df_open_hyde]

indoor = pd.concat(indoor_dfs).sample(2183).reset_index(drop=True)
intermediate = df_inter.sample(1829).reset_index(drop=True)
urban = pd.concat(urban_dfs).sample(2108).reset_index(drop=True)
open_sky = pd.concat(open_dfs).sample(2709).reset_index(drop=True)

given_data = pd.concat([indoor, intermediate, urban, open_sky]).sample(frac=1).reset_index(drop=True)

given_data['num_sat_weak'] = given_data['num_sat'] - given_data['num_sat_25']
given_data['sum_snr_weak'] = given_data['sum_snr'] - given_data['sum_snr_25']

given_data['elev_0_30_weak'] = given_data['elev_0_30'] - given_data['elev_0_30_25']
given_data['sum_elev_0_30_weak'] = given_data['sum_elev_0_30'] - given_data['sum_elev_0_30_25']

given_data['elev_30_60_weak'] = given_data['elev_30_60'] - given_data['elev_30_60_25']
given_data['sum_elev_30_60_weak'] = given_data['sum_elev_30_60'] - given_data['sum_elev_30_60_25']

given_data['elev_60_90_weak'] = given_data['elev_60_90'] - given_data['elev_60_90_25']
given_data['sum_elev_60_90_weak'] = given_data['sum_elev_60_90'] - given_data['sum_elev_60_90_25']

with open('..\saves\given_data_test_set.pickle', 'wb') as f:
    pickle.dump(given_data, f)