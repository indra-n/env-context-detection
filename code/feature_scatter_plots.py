import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


cols=['num_sat', 'sum_snr', 'num_sat_25', 'sum_snr_25', 'elev_0_30', 'elev_30_60', 'elev_60_90',
         'elev_0_30_25', 'elev_30_60_25', 'elev_60_90_25']

with open('..\saves\\exp_train_trim_1.pickle', 'rb') as f:
    train_df = pickle.load(f)

with open('..\saves\\exp_test_trim_1.pickle', 'rb') as f:
    test_df = pickle.load(f)

train_data = train_df
test_data = test_df


indoor = train_data[train_data['true_class'] == 1]
inter = train_data[train_data['true_class'] == 2]
urban = train_data[train_data['true_class'] == 3]
open_sky = train_data[train_data['true_class'] == 4]

print(indoor.shape)
print(inter.shape)
print(urban.shape)
print(open_sky.shape)

by_window = indoor[indoor['location_id']==534].sort_values(by='e_id', ascending=1)
open_side = inter[inter['location_id']==1114].sort_values(by='e_id', ascending=1)


plt.figure(1)
plt.xlabel('Time', fontsize=10)
plt.ylabel('# satelites 25', fontsize=10)
plt.scatter(indoor['e_id'], indoor['num_sat_25'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['num_sat_25'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['num_sat_25'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['num_sat_25'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 20])
# plt.show()

plt.figure(2)
plt.xlabel('Time', fontsize=10)
plt.ylabel('sum SNR 25', fontsize=10)
plt.scatter(indoor['e_id'], indoor['sum_snr_25'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['sum_snr_25'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['sum_snr_25'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['sum_snr_25'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 600])


plt.figure(3)
plt.xlabel('Time', fontsize=10)
plt.ylabel('total # of satelites', fontsize=10)
plt.scatter(indoor['e_id'], indoor['num_sat'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['num_sat'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['num_sat'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['num_sat'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 20])

plt.figure(4)
plt.xlabel('Time', fontsize=10)
plt.ylabel('total sum SNR', fontsize=10)
plt.scatter(indoor['e_id'], indoor['sum_snr'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['sum_snr'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['sum_snr'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['sum_snr'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 300])


plt.figure(5)
plt.xlabel('Time', fontsize=10)
plt.ylabel('# satelites 25dB-Hz elev 60_90', fontsize=10)
plt.scatter(indoor['e_id'], indoor['elev_60_90_25'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['elev_60_90_25'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['elev_60_90_25'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['elev_60_90_25'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 5])

plt.figure(6)
plt.xlabel('Time', fontsize=10)
plt.ylabel('sum SNR 25 dB-Hz  elev 60_90', fontsize=10)
plt.scatter(indoor['e_id'], indoor['sum_elev_60_90_25'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['sum_elev_60_90_25'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['sum_elev_60_90_25'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['sum_elev_60_90_25'], label="open", c='y')
plt.grid(True)
plt.legend()

plt.figure(7)
plt.xlabel('Time', fontsize=10)
plt.ylabel('# satelites 25dB-Hz elev under 30', fontsize=10)
plt.scatter(indoor['e_id'], indoor['elev_0_30_25'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['elev_0_30_25'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['elev_0_30_25'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['elev_0_30_25'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 5])

plt.figure(8)
plt.xlabel('Time', fontsize=10)
plt.ylabel('sum SNR 25 dB-Hz  elev under 30', fontsize=10)
plt.scatter(indoor['e_id'], indoor['sum_elev_0_30_25'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['sum_elev_0_30_25'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['sum_elev_0_30_25'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['sum_elev_0_30_25'], label="open", c='y')
plt.legend()
# plt.axis([0, 350, 0, 150])
plt.grid(True)

plt.figure(9)
plt.xlabel('Time', fontsize=10)
plt.ylabel('# satelites 25dB-Hz elev 30_60', fontsize=10)
plt.scatter(indoor['e_id'], indoor['elev_30_60_25'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['elev_30_60_25'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['elev_30_60_25'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['elev_30_60_25'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 5])

plt.figure(10)
plt.xlabel('Time', fontsize=10)
plt.ylabel('sum SNR 25 dB-Hz  elev 30_60', fontsize=10)
plt.scatter(indoor['e_id'], indoor['sum_elev_30_60_25'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['sum_elev_30_60_25'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['sum_elev_30_60_25'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['sum_elev_30_60_25'], label="open", c='y')
plt.legend()
# plt.axis([0, 350, 0, 150])
plt.grid(True)


plt.figure(11)
plt.xlabel('Time', fontsize=10)
plt.ylabel('# satelites elev 60_90', fontsize=10)
plt.scatter(indoor['e_id'], indoor['elev_60_90'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['elev_60_90'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['elev_60_90'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['elev_60_90'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 5])

plt.figure(12)
plt.xlabel('Time', fontsize=10)
plt.ylabel('sum SNR elev 60_90', fontsize=10)
plt.scatter(indoor['e_id'], indoor['sum_elev_60_90'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['sum_elev_60_90'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['sum_elev_60_90'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['sum_elev_60_90'], label="open", c='y')
plt.grid(True)
plt.legend()

plt.figure(13)
plt.xlabel('Time', fontsize=10)
plt.ylabel('# satelites elev under 30', fontsize=10)
plt.scatter(indoor['e_id'], indoor['elev_0_30'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['elev_0_30'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['elev_0_30'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['elev_0_30'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 5])

plt.figure(14)
plt.xlabel('Time', fontsize=10)
plt.ylabel('sum SNR  elev under 30', fontsize=10)
plt.scatter(indoor['e_id'], indoor['sum_elev_0_30'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['sum_elev_0_30'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['sum_elev_0_30'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['sum_elev_0_30'], label="open", c='y')
plt.legend()
# plt.axis([0, 350, 0, 150])
plt.grid(True)

plt.figure(15)
plt.xlabel('Time', fontsize=10)
plt.ylabel('# satelites elev 30_60', fontsize=10)
plt.scatter(indoor['e_id'], indoor['elev_30_60'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['elev_30_60'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['elev_30_60'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['elev_30_60'], label="open", c='y')
plt.grid(True)
plt.legend()
# plt.axis([0, 350, 0, 5])

plt.figure(16)
plt.xlabel('Time', fontsize=10)
plt.ylabel('sum SNR  elev 30_60', fontsize=10)
plt.scatter(indoor['e_id'], indoor['sum_elev_30_60'], label="indoor", c='r')
plt.scatter(inter['e_id'], inter['sum_elev_30_60'], label="inter", c='b')
plt.scatter(urban['e_id'], urban['sum_elev_30_60'], label="urban", c='g')
plt.scatter(open_sky['e_id'], open_sky['sum_elev_30_60'], label="open", c='y')
plt.legend()
# plt.axis([0, 350, 0, 150])
plt.grid(True)

plt.figure()
plt.xlabel('Time', fontsize=10)
plt.ylabel('sum elev 0_30', fontsize=10)
plt.plot(by_window['e_id'], by_window['sum_elev_30_60'], label="indoor", c='r')
plt.plot(open_side['e_id'], open_side['sum_elev_30_60'], label="inter", c='b')

plt.legend()
# plt.axis([0, 350, 0, 150])
plt.grid(True)

plt.show()

