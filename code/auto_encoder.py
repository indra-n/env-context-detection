import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import math


#Load the data
fp_indoor_bm = '..\data\indoor\British museum\Obs_features'
fp_indoor_ch2221 = '..\data\indoor\Chadwick 2221\Obs_features'
fp_indoor_ch103a = '..\data\indoor\Chadwick 103 by the window\A\Obs_features'
fp_indoor_ch103b = '..\data\indoor\Chadwick 103 by the window\B\Obs_features'
fp_indoor_jah = '..\data\indoor\JAH363A\Obs_features'

fp_inter = '..\data\intermediate\P1\Obs_features'

fp_urban_p1b = '..\data\\urban\P1B\Obs_features'
fp_urban_p2b = '..\data\\urban\P2B\Obs_features'
fp_urban_p3b = '..\data\\urban\P3B\Obs_features'
fp_urban_p4b = '..\data\\urban\P4B\Obs_features'

fp_open_reg = '..\data\open-sky\P1_REGENT\Obs_features'
fp_open_hyde = '..\data\open-sky\P4_HYDE\Obs_features'

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

#cols = ['obs_id', 'e_id', 'sv_prn', 'constell_id', 'azimuth', 'elevation', 'CN0']
# cols = ['sv_prn', 'constell_id', 'azimuth', 'elevation', 'CN0']
# cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25','snr_std', 'elev_0_30', 'elev_30_60', 'elev_60_90',
#          'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']
# cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25','snr_std',
#          'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25', 'frac_30']

# Split training and test data
train_indoor_bm = df_indoor_bm.sample(150)
train_indoor_ch2221 = df_indoor_ch2221.sample(150)
train_indoor_ch103a = df_indoor_ch103a.sample(150)
train_indoor_jah = df_indoor_jah.sample(100)

test_indoor_bm = df_indoor_bm.drop(train_indoor_bm.index).sample(200)
test_indoor_ch2221 = df_indoor_ch2221.drop(train_indoor_ch2221.index).sample(200)
test_indoor_ch103a = df_indoor_ch103a.drop(train_indoor_ch103a.index).sample(200)
test_indoor_ch103b = df_indoor_ch103b.sample(200)
test_indoor_jah = df_indoor_jah.drop(train_indoor_jah.index).sample(200)

train_inter = df_inter.sample(300)
test_inter = df_inter.drop(train_inter.index).sample(400)

train_urban_p1b = df_urban_p1b.sample(150)
train_urban_p2b = df_urban_p2b.sample(150)
train_urban_p4b = df_urban_p4b.sample(150)

test_urban_p1b = df_urban_p1b.drop(train_urban_p1b.index).sample(200)
test_urban_p2b = df_urban_p2b.drop(train_urban_p2b.index).sample(200)
test_urban_p3b = df_urban_p3b.sample(400)
test_urban_p4b = df_urban_p4b.drop(train_urban_p4b.index).sample(200)

train_open_reg = df_open_reg.sample(150)
test_open_hyde = df_open_hyde.sample(600)

train_df = [train_indoor_bm, train_indoor_ch2221, train_indoor_ch103a, train_indoor_jah, train_inter, train_urban_p1b,
            train_urban_p2b, train_urban_p4b, train_open_reg]
train_data = pd.concat(train_df).sample(frac=1).reset_index(drop=True)

test_df = [test_indoor_bm, test_indoor_ch2221, test_indoor_ch103a, test_indoor_ch103b, test_indoor_jah, test_inter,
           test_urban_p1b, test_urban_p2b, test_urban_p3b, test_urban_p4b, test_open_hyde]
test_data = pd.concat(test_df).sample(frac=1).reset_index(drop=True)

full_df = [train_indoor_bm, train_indoor_ch2221, train_indoor_ch103a, train_indoor_jah, train_inter, train_urban_p1b,
            train_urban_p2b, train_urban_p4b, train_open_reg, test_indoor_bm, test_indoor_ch2221, test_indoor_ch103a,
           test_indoor_ch103b, test_indoor_jah, test_inter,
           test_urban_p1b, test_urban_p2b, test_urban_p3b, test_urban_p4b, test_open_hyde]

full_data = pd.concat(test_df).sample(frac=1).reset_index(drop=True)

print(train_data.shape)
print(test_data.shape)

# cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25','snr_std', 'elev_0_30', 'elev_30_60', 'elev_60_90',
#          'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']
# cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25',
#          'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']
cols=['elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']


# Greenwich Data
#Load the data
fp_indoor_cutsark_in1_sp = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\Obs_features_1'
fp_indoor_cutsark_in2_sp = '..\data_Greenwich\indoor\CuttySark_front\CuttySark_front_P2_(inside)\Obs_features_2'
fp_indoor_market_gr1_sp = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\Obs_features_1'
fp_indoor_market_gr2_sp = '..\data_Greenwich\indoor\GreenwichMarket\\under_glass_roof_P2\Obs_features_2'
fp_indoor_museum_gr1_sp = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\Obs_features_1'
fp_indoor_museum_gr2_sp = '..\data_Greenwich\indoor\MaritimeMuseum\hall_underGlassRoof\Obs_features_2'
fp_indoor_museum_lw1_sp = '..\data_Greenwich\indoor\MaritimeMuseum\\under_light_well\Obs_features_1'
fp_indoor_museum_lw2_sp = '..\data_Greenwich\indoor\MaritimeMuseum\\under_light_well\Obs_features_2'

fp_inter_path1_sp = '..\data_Greenwich\intermediate\covered_path_byGym\Obs_features_1'
fp_inter_path2_sp = '..\data_Greenwich\intermediate\covered_path_byGym\Obs_features_2'
fp_inter_dept3_sp = '..\data_Greenwich\intermediate\Deptford_TrainStation\P3\Obs_features'
fp_inter_GreenTS_p1_1_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\Obs_features_1'
fp_inter_GreenTS_p1_2_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P1\Obs_features_2'
fp_inter_GreenTS_p2_1_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\Obs_features_1'
fp_inter_GreenTS_p2_2_sp = '..\data_Greenwich\intermediate\Greenwich_TrainStation\P2\Obs_features_2'
fp_inter_market_aw1_sp = '..\data_Greenwich\intermediate\GreenwichMarket\entrance_archway_P1\Obs_features_1'
fp_inter_market_aw2_sp = '..\data_Greenwich\intermediate\GreenwichMarket\entrance_archway_P1\Obs_features_2'
fp_inter_park_dark1_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\Obs_features_1'
fp_inter_park_dark2_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_dark\Obs_features_2'
fp_inter_park_light1_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_lighter\Obs_features_1'
fp_inter_park_light2_sp = '..\data_Greenwich\intermediate\GreenwichPark\\tree_cover_lighter\Obs_features_2'
fp_inter_queens_arch_sp = '..\data_Greenwich\intermediate\QueensHouse\\archway\Obs_features'
fp_inter_queens_col1_sp = '..\data_Greenwich\intermediate\QueensHouse\colonnade\Obs_features_1'
fp_inter_queens_col2_sp = '..\data_Greenwich\intermediate\QueensHouse\colonnade\Obs_features_2'

fp_open_park1_sp = '..\data_Greenwich\open_sky\GreenwichPark\open\Obs_features_1'
fp_open_park2_sp = '..\data_Greenwich\open_sky\GreenwichPark\open\Obs_features_2'

fp_urban_sl1_sp = '..\data_Greenwich\\urban\\behind_SailLoftPub\Obs_features_1'
fp_urban_sl2_sp = '..\data_Greenwich\\urban\\behind_SailLoftPub\Obs_features_2'
fp_urban_cutsark_out1_sp = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\Obs_features_1'
fp_urban_cutsark_out2_sp = '..\data_Greenwich\\urban\CuttySark_front\CuttySark_front_P1_(outside)\Obs_features_2'
fp_urban_dept1_sp = '..\data_Greenwich\\urban\Deptford_TrainStation\P1\Obs_features'
fp_urban_dept2_sp = '..\data_Greenwich\\urban\Deptford_TrainStation\P2\Obs_features'
fp_urban_GreenTS_p3_1_sp = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\Obs_features_1'
fp_urban_GreenTS_p3_2_sp = '..\data_Greenwich\\urban\Greenwich_TrainStation\P3\Obs_features_2'
fp_urban_queens_court_sp = '..\data_Greenwich\\urban\QueensHouse\courtyard\Obs_features'


# Load in dataframe
df_indoor_cutsark_in1 = pd.read_csv(fp_indoor_cutsark_in1_sp)
df_indoor_cutsark_in2 = pd.read_csv(fp_indoor_cutsark_in2_sp)
df_indoor_market_gr1 = pd.read_csv(fp_indoor_market_gr1_sp)
df_indoor_market_gr2 = pd.read_csv(fp_indoor_market_gr2_sp)
df_indoor_museum_gr1 = pd.read_csv(fp_indoor_museum_gr1_sp)
df_indoor_museum_gr2 = pd.read_csv(fp_indoor_museum_gr2_sp)
df_indoor_museum_lw1 = pd.read_csv(fp_indoor_museum_lw1_sp)
df_indoor_museum_lw2 = pd.read_csv(fp_indoor_museum_lw2_sp)

df_indoor_cutsark_in1['true_class'] = 1
df_indoor_cutsark_in2['true_class'] = 1
df_indoor_market_gr1['true_class'] = 1
df_indoor_market_gr2['true_class'] = 1
df_indoor_museum_gr1['true_class'] = 1
df_indoor_museum_gr2['true_class'] = 1
df_indoor_museum_lw1['true_class'] = 1
df_indoor_museum_lw2['true_class'] = 1

df_indoor_cutsark_in1['location'] = 321
df_indoor_cutsark_in2['location'] = 322
df_indoor_market_gr1['location'] = 421
df_indoor_market_gr2['location'] = 422
df_indoor_museum_gr1['location'] = 511
df_indoor_museum_gr2['location'] = 512
df_indoor_museum_lw1['location'] = 521
df_indoor_museum_lw2['location'] = 522


df_inter_path1 = pd.read_csv(fp_inter_path1_sp)
df_inter_path2 = pd.read_csv(fp_inter_path2_sp)
df_inter_dept3 = pd.read_csv(fp_inter_dept3_sp)
df_inter_GreenTS_p1_1 = pd.read_csv(fp_inter_GreenTS_p1_1_sp)
df_inter_GreenTS_p1_2 = pd.read_csv(fp_inter_GreenTS_p1_2_sp)
df_inter_GreenTS_p2_1 = pd.read_csv(fp_inter_GreenTS_p2_1_sp)
df_inter_GreenTS_p2_2 = pd.read_csv(fp_inter_GreenTS_p2_2_sp)
df_inter_market_aw1 = pd.read_csv(fp_inter_market_aw1_sp)
df_inter_market_aw2 = pd.read_csv(fp_inter_market_aw2_sp)
df_inter_park_dark1 = pd.read_csv(fp_inter_park_dark1_sp)
df_inter_park_dark2 = pd.read_csv(fp_inter_park_dark2_sp)
df_inter_park_light1 = pd.read_csv(fp_inter_park_light1_sp)
df_inter_park_light2 = pd.read_csv(fp_inter_park_light2_sp)
df_inter_queens_arch = pd.read_csv(fp_inter_queens_arch_sp)
df_inter_queens_col1 = pd.read_csv(fp_inter_queens_col1_sp)
df_inter_queens_col2 = pd.read_csv(fp_inter_queens_col2_sp)

df_inter_path1['location'] = 211
df_inter_path2['location'] = 212
df_inter_dept3['location'] = 931
df_inter_GreenTS_p1_1['location'] = 811
df_inter_GreenTS_p1_2['location'] = 812
df_inter_GreenTS_p2_1['location'] = 821
df_inter_GreenTS_p2_2['location'] = 822
df_inter_market_aw1['location'] = 411
df_inter_market_aw2['location'] = 412
df_inter_park_dark1['location'] = 721
df_inter_park_dark2['location'] = 722
df_inter_park_light1['location'] = 731
df_inter_park_light2['location'] = 732
df_inter_queens_arch['location'] = 631
df_inter_queens_col1['location'] = 611
df_inter_queens_col2['location'] = 612

df_inter_path1['true_class'] = 2
df_inter_path2['true_class'] = 2
df_inter_dept3['true_class'] = 2
df_inter_GreenTS_p1_1['true_class'] = 2
df_inter_GreenTS_p1_2['true_class'] = 2
df_inter_GreenTS_p2_1['true_class'] = 2
df_inter_GreenTS_p2_2['true_class'] = 2
df_inter_market_aw1['true_class'] = 2
df_inter_market_aw2['true_class'] = 2
df_inter_park_dark1['true_class'] = 2
df_inter_park_dark2['true_class'] = 2
df_inter_park_light1['true_class'] = 2
df_inter_park_light2['true_class'] = 2
df_inter_queens_arch['true_class'] = 2
df_inter_queens_col1['true_class'] = 2
df_inter_queens_col2['true_class'] = 2


df_open_park1 = pd.read_csv(fp_open_park1_sp)
df_open_park2 = pd.read_csv(fp_open_park2_sp)

df_open_park1['true_class'] = 4
df_open_park2['true_class'] = 4

df_open_park1['location'] = 711
df_open_park2['location'] = 712


df_urban_sl1 = pd.read_csv(fp_urban_sl1_sp)
df_urban_sl2 = pd.read_csv(fp_urban_sl2_sp)
df_urban_cutsark_out1 = pd.read_csv(fp_urban_cutsark_out1_sp)
df_urban_cutsark_out2 = pd.read_csv(fp_urban_cutsark_out2_sp)
df_urban_dept1 = pd.read_csv(fp_urban_dept1_sp)
df_urban_dept2 = pd.read_csv(fp_urban_dept2_sp)
df_urban_GreenTS_p3_1 = pd.read_csv(fp_urban_GreenTS_p3_1_sp)
df_urban_GreenTS_p3_2 = pd.read_csv(fp_urban_GreenTS_p3_2_sp)
df_urban_queens_court = pd.read_csv(fp_urban_queens_court_sp)

df_urban_sl1['true_class'] = 3
df_urban_sl2['true_class'] = 3
df_urban_cutsark_out1['true_class'] = 3
df_urban_cutsark_out2['true_class'] = 3
df_urban_dept1['true_class'] = 3
df_urban_dept2['true_class'] = 3
df_urban_GreenTS_p3_1['true_class'] = 3
df_urban_GreenTS_p3_2['true_class'] = 3
df_urban_queens_court['true_class'] = 3

df_urban_sl1['location'] = 111
df_urban_sl2['location'] = 112
df_urban_cutsark_out1['location'] = 311
df_urban_cutsark_out2['location'] = 312
df_urban_dept1['location'] = 911
df_urban_dept2['location'] = 921
df_urban_GreenTS_p3_1['location'] = 831
df_urban_GreenTS_p3_2['location'] = 832
df_urban_queens_court['location'] = 621

########
train_indoor_1 = df_indoor_cutsark_in1.sample(60)
test_indoor_1 = df_indoor_cutsark_in2.sample(60)
train_indoor_2 = df_indoor_market_gr1.sample(40)
test_indoor_2 = df_indoor_market_gr2.sample(60)
train_indoor_3 = df_indoor_museum_gr1.sample(60)
test_indoor_3 = df_indoor_museum_gr2.sample(60)
train_indoor_4 = df_indoor_museum_lw1.sample(30)
test_indoor_4 = df_indoor_museum_lw2.sample(15)

train_inter_1 = df_inter_path1.sample(40)
test_inter_1 = df_inter_path2.sample(60)
test_inter_2 = df_inter_dept3.sample(60)
train_inter_2 = df_inter_GreenTS_p1_1.sample(60)
test_inter_3 = df_inter_GreenTS_p1_2.sample(60)
train_inter_3 = df_inter_GreenTS_p2_1.sample(60)
test_inter_4 = df_inter_GreenTS_p2_2.sample(60)
train_inter_4 = df_inter_market_aw1.sample(40)
test_inter_5 = df_inter_market_aw2.sample(60)
train_inter_5 = df_inter_park_dark1.sample(40)
test_inter_6 = df_inter_park_dark2.sample(60)
train_inter_6 = df_inter_park_light1.sample(60)
test_inter_9 = df_inter_park_light2.sample(60)
test_inter_7 = df_inter_queens_arch.sample(60)
train_inter_7 = df_inter_queens_col1.sample(60)
test_inter_8 = df_inter_queens_col2.sample(60)

train_urban_1 = df_urban_sl1.sample(60)
test_urban_1 = df_urban_sl2.sample(60)
train_urban_2 = df_urban_cutsark_out1.sample(40)
test_urban_2 = df_urban_cutsark_out2.sample(60)
train_urban_3 = df_urban_dept1.sample(60)
test_urban_3 = df_urban_dept2.sample(60)
train_urban_4 = df_urban_GreenTS_p3_1.sample(40)
test_urban_4 = df_urban_GreenTS_p3_2.sample(60)
train_urban_5 = df_urban_queens_court.sample(60)

train_open = df_open_park1.sample(60)
test_open = df_open_park2.sample(60)
######

greenwich_df = [train_indoor_1, train_indoor_2, train_indoor_3, train_indoor_4, train_inter_1, train_inter_2, train_inter_3,
            train_inter_4, train_inter_5, train_inter_6, train_inter_7, train_urban_1, train_urban_2, train_urban_3,
            train_urban_4, train_urban_5, train_open, test_indoor_1, test_indoor_2, test_indoor_3, test_indoor_4,
            test_inter_1, test_inter_2, test_inter_3,
           test_inter_4, test_inter_5, test_inter_6, test_inter_7, test_inter_8, test_inter_9, test_urban_1,
           test_urban_2, test_urban_3, test_urban_4, test_open]
greenwich_data = pd.concat(greenwich_df).sample(frac=1).reset_index(drop=True)



h_dim = 2
input_dim = len(cols)
print(input_dim)

x = tf.placeholder(tf.float32, [None, input_dim])
#    y_true = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal((input_dim, h_dim))) #Should not be 0
b = tf.Variable(tf.zeros([h_dim]))

h = tf.nn.tanh(tf.matmul(x, W) + b)

W_o = tf.transpose(W)
b_o = tf.Variable(tf.zeros([input_dim]))

y_ = tf.nn.tanh(tf.matmul(h, W_o) + b_o)
y_true = tf.placeholder("float", [None, input_dim])

meansq = tf.reduce_mean(tf.square(y_-y_true))
train_step = tf.train.AdamOptimizer(0.005).minimize(meansq)


#    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
valid_set = test_data.sample(1000)
valid_set['log_sum_25'] = np.nan_to_num(np.log(valid_set['sum_snr_25']))
train_data['log_sum_25'] = np.nan_to_num(np.log(train_data['sum_snr_25']))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch = train_data.sample(256)[cols]
        # for batch in np.array_split(train_data[cols], 55):
        #     for batch in np.array_split(full_data[cols], 89):
        err, _ = sess.run([meansq, train_step], feed_dict={x: batch, y_true: batch})
        if i % 100 == 0:
            err, _ = sess.run([meansq, train_step], feed_dict={x: valid_set[cols], y_true: valid_set[cols]})
            print(err)
    # for i in range(18):
    #     batch_x = train_data.
    #     sess.run(train_step, feed_dict={x: batch_x, y_true: batch_y})
    # test_score = accuracy.eval(feed_dict={x: data.test.images, y_true: data.test.labels})

    compressed = h.eval(feed_dict={x: test_data[cols], y_true: test_data[cols]})

    # saver = tf.train.Saver()
    # saver.save(sess, "/save_points/ae_model_full.ckpt")
    #
    sess.close()

# saver = tf.train.Saver()


# with tf.Session() as sess:
#     # Restore variables from disk.
#     saver.restore(sess, "/save_points/ae_model_full.ckpt")
#     print("Model restored.")
#     compressed = h.eval(feed_dict={x: greenwich_data[cols], y_true: greenwich_data[cols]})

with open('orig_data_ae_3col.pickle', 'wb') as f:
    pickle.dump(test_data, f)

with open('compressed_data_ae2_3col.pickle', 'wb') as f:
    pickle.dump(compressed, f)

# with open('ae_model.pickle', 'wb') as f:
#     pickle.dump(compressed, f)

# with open('orig_green_data_ae.pickle', 'wb') as f:
#     pickle.dump(greenwich_data, f)
#
# with open('compressed_green_data_ae.pickle', 'wb') as f:
#     pickle.dump(compressed, f)

