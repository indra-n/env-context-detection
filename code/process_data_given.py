import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
import csv

# Load the data

fp_indoor_bm = '..\data\indoor\British museum\Observations'
fp_indoor_ch2221 = '..\data\indoor\Chadwick 2221\Observations'
fp_indoor_ch103a = '..\data\indoor\Chadwick 103 by the window\A\Observations'
fp_indoor_ch103b = '..\data\indoor\Chadwick 103 by the window\B\Observations'
fp_indoor_jah = '..\data\indoor\JAH363A\Observations'

fp_inter = '..\data\intermediate\P1\Observations'

fp_urban_p1b = '..\data\\urban\P1B\Observations.csv'
fp_urban_p2b = '..\data\\urban\P2B\Observations.csv'
fp_urban_p3b = '..\data\\urban\P3B\Observations.csv'
fp_urban_p4b = '..\data\\urban\P4B\Observations.csv'

fp_open_reg = '..\data\open-sky\P1_REGENT\Observations'
fp_open_hyde = '..\data\open-sky\P4_HYDE\Observations'

######
fp_indoor_bm_sp = '..\data\indoor\British museum\Obs_features_ext'
fp_indoor_ch2221_sp = '..\data\indoor\Chadwick 2221\Obs_features_ext'
fp_indoor_ch103a_sp = '..\data\indoor\Chadwick 103 by the window\A\Obs_features_ext'
fp_indoor_ch103b_sp = '..\data\indoor\Chadwick 103 by the window\B\Obs_features_ext'
fp_indoor_jah_sp = '..\data\indoor\JAH363A\Obs_features_ext'

fp_inter_sp = '..\data\intermediate\P1\Obs_features_ext'

fp_urban_p1b_sp = '..\data\\urban\P1B\Obs_features_ext'
fp_urban_p2b_sp = '..\data\\urban\P2B\Obs_features_ext'
fp_urban_p3b_sp = '..\data\\urban\P3B\Obs_features_ext'
fp_urban_p4b_sp = '..\data\\urban\P4B\Obs_features_ext'

fp_open_reg_sp = '..\data\open-sky\P1_REGENT\Obs_features_ext'
fp_open_hyde_sp = '..\data\open-sky\P4_HYDE\Obs_features_ext'




def create_feature_file(data_path, save_path):
    df_data = pd.read_csv(data_path)

    # Clean data by deleting entries with 0 values
    df_clean = df_data.query('(CN0 != 0.0) & (elevation != 0.0)')

    # Generate features
    grouped = df_clean.groupby('e_id')
    feature_df = pd.DataFrame(index=grouped.indices, columns=['e_id', 'num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25',
                                                              'num_sat_u20', 'sum_sat_u20', 'snr_std',
                                                              'elev_0_30', 'elev_30_60', 'elev_60_90',
                                                              'sum_elev_0_30', 'sum_elev_30_60', 'sum_elev_60_90',
                                                              'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25',
                                                              'sum_elev_0_30_25', 'sum_elev_30_60_25', 'sum_elev_60_90_25'])

    for epoch_id, group in grouped:
        over_thresh = group[group.CN0 >= 25.0]
        under_thresh = group[group.CN0 <= 20.0]
        feature_df.loc[epoch_id] = [epoch_id, group.shape[0], group.CN0.sum(), over_thresh.shape[0], over_thresh.CN0.sum(),
                                    under_thresh.shape[0], under_thresh.CN0.sum(), group.CN0.std(),
                                    group[group.elevation <= 30.0].shape[0],
                                    group[(group.elevation <= 60.0) & (group.elevation > 30.0)].shape[0],
                                    group[group.elevation > 60.0].shape[0],
                                    group[group.elevation <= 30.0].CN0.sum(),
                                    group[(group.elevation <= 60.0) & (group.elevation > 30.0)].CN0.sum(),
                                    group[group.elevation > 60.0].CN0.sum(),
                                    over_thresh[over_thresh.elevation <= 30.0].shape[0],
                                    over_thresh[(over_thresh.elevation <= 60.0) & (over_thresh.elevation > 30.0)].shape[0],
                                    over_thresh[over_thresh.elevation > 60.0].shape[0],
                                    over_thresh[over_thresh.elevation <= 30.0].CN0.sum(),
                                    over_thresh[(over_thresh.elevation <= 60.0) & (over_thresh.elevation > 30.0)].CN0.sum(),
                                    over_thresh[over_thresh.elevation > 60.0].CN0.sum()
        ]
    feature_df.to_csv(save_path)


# Create feature files
#
create_feature_file(fp_indoor_bm, fp_indoor_bm_sp)
create_feature_file(fp_indoor_ch2221, fp_indoor_ch2221_sp)
create_feature_file(fp_indoor_ch103a, fp_indoor_ch103a_sp)
create_feature_file(fp_indoor_ch103b, fp_indoor_ch103b_sp)
create_feature_file(fp_indoor_jah, fp_indoor_jah_sp)

create_feature_file(fp_inter, fp_inter_sp)

create_feature_file(fp_urban_p1b, fp_urban_p1b_sp)
create_feature_file(fp_urban_p2b, fp_urban_p2b_sp)
create_feature_file(fp_urban_p3b, fp_urban_p3b_sp)
create_feature_file(fp_urban_p4b, fp_urban_p4b_sp)

create_feature_file(fp_open_reg, fp_open_reg_sp)
create_feature_file(fp_open_hyde, fp_open_hyde_sp)
