# @Time : 2022/2/17 17:06 
# @Author : zhongyu 
# @File : data_process.py


import numpy as np
import h5py
import hdf5Reader2A as h5r
from scipy import signal
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import math
import file_read as fr
from Feature_extraction import factors_profile
import Feature_extraction as feaex


def error_del(shotset, errorset):
    r_del = []
    for j in range(shotset.shape[0]):
        if shotset[j, 0] in errorset:
            r_del.append(j)
    finalset = np.delete(shotset, r_del, axis=0)
    return finalset


def profile_info(shot, array_t, start, end):
    # time, data = h5r.read_channel(shot, channel=array_t[0], device="2a")
    # time_ip, data_ip = h5r.read_channel(shot, channel='IP', device="2a")
    time, data = fr.read_data(shot, array_t[0], start, end)
    array_1 = np.empty([len(array_t), data.shape[0]])
    for i in range(len(array_t)):
        # time, data = h5r.read_channel(shot, channel=array_t[i], device="2a")
        time, data = fr.read_data(shot, array_t[i], start, end)
        array_1[i, :] = data
    skew, kurt, var, time_down = factors_profile(array_1, time, len(array_1), 1000)

    # start = np.where(time_ip > start)[0][0]
    # end = np.where(time_ip >= end)[0][0]
    # skew = skew[start:end]
    # kurt = kurt[start:end]
    # var = var[start:end]
    # time_down = time_ip[start:end]
    return skew, kurt, var, time_down


def shot_data(shot, disr_tag, channels_down, channels, endtime, col_num):
    # 每一炮的开始和结束时间
    start = h5r.get_attrs("StartTime", shot_number=shot, channel="EFIT_LI")
    end = endtime / 1000

    # 计算Δip
    time1, data_ip = fr.read_data(shot, "IP", start, end)
    time2, data_tip = fr.read_data(shot, "IP_TARGET", start, end)
    delta_ip = (data_ip - data_tip) / data_ip
    data_matrix = np.zeros((len(delta_ip), col_num))
    # data_matrix = np.zeros((len(delta_ip) - 30 + 2, col_num))  # 贴合加窗后的fft信号
    data_matrix[:, 0] = delta_ip

    # 计算βN
    time3, data_betaT = fr.read_data(shot, "EFIT_BETA_T", start, end)
    time4, data_bt = fr.read_data(shot, "BT", start, end)
    time5, data_r = fr.read_data(shot, "EFIT_MINOR_R", start, end)
    betaN = 1000 * data_betaT * 0.0622 * data_r * data_bt / data_ip
    data_matrix[:, 1] = betaN

    # 降采样：对需要降采样的信号
    i = 2
    for channel in channels_down:
        time, data = fr.read_data(shot, channel, start, end)
        data1 = signal.resample(data, num=len(delta_ip), t=time, axis=0)
        data_matrix[:, i] = data1[0]
        i += 1

    # 不需要降采样的信号直接读取
    for channel in channels:
        time, data = fr.read_data(shot, channel, start, end)
        data_matrix[:, i] = data
        i += 1

    # profile
    str1 = ['01', '02', '03', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20']
    strsx = ['SX' + j for j in str1]
    str2 = ['03', '04', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
    strbolu = ['BOLU' + j for j in str2]
    str3 = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
    strbold = ['BOLD' + j for j in str3]
    skewsx, kurtsx, varsx, time_downsx = profile_info(shot, strsx, start, end)
    data_matrix[:, i] = skewsx
    i += 1
    data_matrix[:, i] = kurtsx
    i += 1
    data_matrix[:, i] = varsx
    i += 1
    skewbolu, kurtbolu, varbolu, time_downbolu = profile_info(shot, strbolu, start, end)
    data_matrix[:, i] = skewbolu
    i += 1
    data_matrix[:, i] = kurtbolu
    i += 1
    data_matrix[:, i] = varbolu
    i += 1
    skewbold, kurtbold, varbold, time_downbold = profile_info(shot, strbold, start, end)
    data_matrix[:, i] = skewbold
    i += 1
    data_matrix[:, i] = kurtbold
    i += 1
    data_matrix[:, i] = varbold
    i += 1

    # 是否破裂标签
    if disr_tag == 1:
        data_matrix[-100:, i] = np.ones([1, 100])

    data_matrix[:, i + 1] = np.ones([1, len(delta_ip)]) * shot  # 炮号
    data_matrix[:, i + 2] = time  # 时间
    data_matrix[:, i + 3] = np.ones([1, len(delta_ip)]) * endtime  # 结束时间
    return data_matrix


def shot_data_fft(shot, disr_tag, channels_down, channels, endtime, col_num):
    # 每一炮的开始和结束时间
    start = h5r.get_attrs("StartTime", shot_number=shot, channel="EFIT_LI")
    end = endtime / 1000

    # 计算Δip
    time1, data_ip = fr.read_data(shot, "IP", start, end)
    time2, data_tip = fr.read_data(shot, "IP_TARGET", start, end)
    delta_ip = (data_ip - data_tip) / data_ip
    data_matrix = np.zeros((len(delta_ip) - 30 + 2, col_num))  # 贴合加窗后的fft信号
    data_matrix[:, 0] = delta_ip[28:]

    # 计算βN
    time3, data_betaT = fr.read_data(shot, "EFIT_BETA_T", start, end)
    time4, data_bt = fr.read_data(shot, "BT", start, end)
    time5, data_r = fr.read_data(shot, "EFIT_MINOR_R", start, end)
    betaN = 1000 * data_betaT * 0.0622 * data_r * data_bt / data_ip
    data_matrix[:, 1] = betaN[28:]

    # 降采样：对需要降采样的信号
    i = 2
    for channel in channels_down:
        time, data = fr.read_data(shot, channel, start, end)
        data1 = signal.resample(data, num=len(delta_ip), t=time, axis=0)
        data_matrix[:, i] = data1[0][28:]
        i += 1

    # 不需要降采样的信号直接读取
    for channel in channels:
        time, data = fr.read_data(shot, channel, start, end)
        data_matrix[:, i] = data[28:]
        i += 1

    # profile
    str1 = ['01', '02', '03', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20']
    strsx = ['SX' + j for j in str1]
    str2 = ['03', '04', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
    strbolu = ['BOLU' + j for j in str2]
    str3 = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
    strbold = ['BOLD' + j for j in str3]
    skewsx, kurtsx, varsx, time_downsx = profile_info(shot, strsx, start, end)
    data_matrix[:, i] = skewsx[28:]
    i += 1
    data_matrix[:, i] = kurtsx[28:]
    i += 1
    data_matrix[:, i] = varsx[28:]
    i += 1
    skewbolu, kurtbolu, varbolu, time_downbolu = profile_info(shot, strbolu, start, end)
    data_matrix[:, i] = skewbolu[28:]
    i += 1
    data_matrix[:, i] = kurtbolu[28:]
    i += 1
    data_matrix[:, i] = varbolu[28:]
    i += 1
    skewbold, kurtbold, varbold, time_downbold = profile_info(shot, strbold, start, end)
    data_matrix[:, i] = skewbold[28:]
    i += 1
    data_matrix[:, i] = kurtbold[28:]
    i += 1
    data_matrix[:, i] = varbold[28:]
    i += 1

    # Mirnov signal
    time_m, data_mp04 = fr.read_data(shot, 'MP04', start, end)
    time_m, data_mp13 = fr.read_data(shot, 'MP13', start, end)
    time_n, data_np04 = fr.read_data(shot, 'NP04', start, end)
    time_n, data_np09 = fr.read_data(shot, 'NP09', start, end)
    data_mp04, time_m04, fsm = feaex.time_win(data_mp04, time_m, win_size=0.030, step=0.001)
    data_mp13, time_m13, fsm = feaex.time_win(data_mp13, time_m, win_size=0.030, step=0.001)
    data_np04, time_n04, fsm = feaex.time_win(data_np04, time_n, win_size=0.030, step=0.001)
    data_np09, time_n09, fsm = feaex.time_win(data_np09, time_n, win_size=0.030, step=0.001)
    mp04_fre = feaex.freq_data(data_mp04, fsm)
    mp13_fre = feaex.freq_data(data_mp13, fsm)
    np04_fre = feaex.freq_data(data_np04, fsm)
    np09_fre = feaex.freq_data(data_np09, fsm)
    mp04_fft = feaex.fft_zy(data_mp04, fsm)
    mp13_fft = feaex.fft_zy(data_mp13, fsm)
    np04_fft = feaex.fft_zy(data_np04, fsm)
    np09_fft = feaex.fft_zy(data_np09, fsm)
    data_matrix[:, i:i + 16] = np.array(mp04_fre)
    i += 16
    data_matrix[:, i:i + 2] = np.array(mp04_fft).T
    i += 2
    data_matrix[:, i:i + 16] = np.array(mp13_fre)
    i += 16
    data_matrix[:, i:i + 2] = np.array(mp13_fft).T
    i += 2
    data_matrix[:, i:i + 16] = np.array(np04_fre)
    i += 16
    data_matrix[:, i:i + 2] = np.array(np04_fft).T
    i += 2
    data_matrix[:, i:i + 16] = np.array(np09_fre)
    i += 16
    data_matrix[:, i:i + 2] = np.array(np09_fft).T
    i += 2

    # 是否破裂标签
    if disr_tag == 1:
        data_matrix[-100:, i] = np.ones([1, 100])

    data_matrix[:, i + 1] = np.ones([1, len(delta_ip) - 28]) * shot  # 炮号
    data_matrix[:, i + 2] = time[28:]  # 时间
    data_matrix[:, i + 3] = np.ones([1, len(delta_ip) - 28]) * endtime  # 结束时间
    return data_matrix


def shot_data_win(shot, disr_tag, channels_down, channels, endtime, col_num):
    # 每一炮的开始和结束时间
    start = h5r.get_attrs("StartTime", shot_number=shot, channel="EFIT_LI")
    end = endtime / 1000

    # 计算Δip
    time1, data_ip = fr.read_data(shot, "IP", start, end)
    time2, data_tip = fr.read_data(shot, "IP_TARGET", start, end)
    delta_ip = (data_ip - data_tip) / data_ip
    X_ip, time1, fs1 = feaex.time_win(delta_ip, time1, win_size=0.030, step=0.001)
    X_ip = feaex.mean_win_data(X_ip)
    data_matrix = np.zeros((len(X_ip), col_num))
    data_matrix[:, 0] = X_ip

    # 计算βN
    time3, data_betaT = fr.read_data(shot, "EFIT_BETA_T", start, end)
    time4, data_bt = fr.read_data(shot, "BT", start, end)
    time5, data_r = fr.read_data(shot, "EFIT_MINOR_R", start, end)
    betaN = 1000 * data_betaT * 0.0622 * data_r * data_bt / data_ip
    betaN, time3, fs3 = feaex.time_win(betaN, time3, win_size=0.030, step=0.001)
    betaN = feaex.mean_win_data(betaN)
    data_matrix[:, 1] = betaN

    # 降采样：对需要降采样的信号
    i = 2
    for channel in channels_down:
        time, data = fr.read_data(shot, channel, start, end)
        data_w, time, fs6 = feaex.time_win(data, time, win_size=0.030, step=0.001)
        data_wm = feaex.mean_win_data(data_w)
        data_matrix[:, i] = data_wm
        i += 1

    # 不需要降采样的信号直接读取
    for channel in channels:
        time, data = fr.read_data(shot, channel, start, end)
        data, time, fs6 = feaex.time_win(data, time, win_size=0.030, step=0.001)
        data = feaex.mean_win_data(data)
        data_matrix[:, i] = data
        i += 1

    # profile
    str1 = ['01', '02', '03', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20']
    strsx = ['SX' + j for j in str1]
    str2 = ['03', '04', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
    strbolu = ['BOLU' + j for j in str2]
    str3 = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
    strbold = ['BOLD' + j for j in str3]
    skewsx, kurtsx, varsx, time_downsx = profile_info(shot, strsx, start, end)
    skewsx, time, fs7 = feaex.time_win(skewsx, time_downsx, win_size=0.030, step=0.001)
    skewsx = feaex.mean_win_data(skewsx)
    kurtsx, time, fs7 = feaex.time_win(kurtsx, time_downsx, win_size=0.030, step=0.001)
    kurtsx = feaex.mean_win_data(kurtsx)
    varsx, time, fs7 = feaex.time_win(varsx, time_downsx, win_size=0.030, step=0.001)
    varsx = feaex.mean_win_data(varsx)
    data_matrix[:, i] = skewsx
    i += 1
    data_matrix[:, i] = kurtsx
    i += 1
    data_matrix[:, i] = varsx
    i += 1
    skewbolu, kurtbolu, varbolu, time_downbolu = profile_info(shot, strbolu, start, end)
    skewbolu, time, fs7 = feaex.time_win(skewbolu, time_downbolu, win_size=0.030, step=0.001)
    skewbolu = feaex.mean_win_data(skewbolu)
    kurtbolu, time, fs7 = feaex.time_win(kurtbolu, time_downbolu, win_size=0.030, step=0.001)
    kurtbolu = feaex.mean_win_data(kurtbolu)
    varbolu, time, fs7 = feaex.time_win(varbolu, time_downbolu, win_size=0.030, step=0.001)
    varbolu = feaex.mean_win_data(varbolu)
    data_matrix[:, i] = skewbolu
    i += 1
    data_matrix[:, i] = kurtbolu
    i += 1
    data_matrix[:, i] = varbolu
    i += 1
    skewbold, kurtbold, varbold, time_downbold = profile_info(shot, strbold, start, end)
    skewbold, time, fs7 = feaex.time_win(skewbold, time_downbold, win_size=0.030, step=0.001)
    skewbold = feaex.mean_win_data(skewbold)
    kurtbold, time, fs7 = feaex.time_win(kurtbold, time_downbold, win_size=0.030, step=0.001)
    kurtbold = feaex.mean_win_data(kurtbold)
    varbold, time, fs7 = feaex.time_win(varbold, time_downbold, win_size=0.030, step=0.001)
    varbold = feaex.mean_win_data(varbold)
    data_matrix[:, i] = skewbold
    i += 1
    data_matrix[:, i] = kurtbold
    i += 1
    data_matrix[:, i] = varbold
    i += 1

    # Mirnov signal
    time_m, data_mp04 = fr.read_data(shot, 'MP04', start, end)
    time_m, data_mp13 = fr.read_data(shot, 'MP13', start, end)
    time_n, data_np04 = fr.read_data(shot, 'NP04', start, end)
    time_n, data_np09 = fr.read_data(shot, 'NP09', start, end)
    data_mp04, time_m04, fsm = feaex.time_win(data_mp04, time_m, win_size=0.030, step=0.001)
    data_mp13, time_m13, fsm = feaex.time_win(data_mp13, time_m, win_size=0.030, step=0.001)
    data_np04, time_n04, fsm = feaex.time_win(data_np04, time_n, win_size=0.030, step=0.001)
    data_np09, time_n09, fsm = feaex.time_win(data_np09, time_n, win_size=0.030, step=0.001)
    mp04_fre = feaex.freq_data(data_mp04, fsm)
    mp13_fre = feaex.freq_data(data_mp13, fsm)
    np04_fre = feaex.freq_data(data_np04, fsm)
    np09_fre = feaex.freq_data(data_np09, fsm)
    mp04_fft = feaex.fft_zy(data_mp04, fsm)
    mp13_fft = feaex.fft_zy(data_mp13, fsm)
    np04_fft = feaex.fft_zy(data_np04, fsm)
    np09_fft = feaex.fft_zy(data_np09, fsm)
    data_matrix[:, i:i + 16] = np.array(mp04_fre)
    i += 16
    data_matrix[:, i:i + 2] = np.array(mp04_fft).T
    i += 2
    data_matrix[:, i:i + 16] = np.array(mp13_fre)
    i += 16
    data_matrix[:, i:i + 2] = np.array(mp13_fft).T
    i += 2
    data_matrix[:, i:i + 16] = np.array(np04_fre)
    i += 16
    data_matrix[:, i:i + 2] = np.array(np04_fft).T
    i += 2
    data_matrix[:, i:i + 16] = np.array(np09_fre)
    i += 16
    data_matrix[:, i:i + 2] = np.array(np09_fft).T
    i += 2

    # 是否破裂标签
    if disr_tag == 1:
        data_matrix[-100:, i] = np.ones([1, 100])

    data_matrix[:, i + 1] = np.ones([1, len(X_ip)]) * shot  # 炮号
    data_matrix[:, i + 2] = time  # 时间
    data_matrix[:, i + 3] = np.ones([1, len(X_ip)]) * endtime  # 结束时间
    return data_matrix


def shot_data_win_add(shot, disr_tag, channels_down, channels, endtime, col_num):
    # 每一炮的开始和结束时间
    start = h5r.get_attrs("StartTime", shot_number=shot, channel="EFIT_LI")
    end = endtime / 1000

    # 计算Δip
    time1, data_ip = fr.read_data(shot, "IP", start, end)
    time2, data_tip = fr.read_data(shot, "IP_TARGET", start, end)
    delta_ip = (data_ip - data_tip) / data_ip
    X_ip, time1, fs1 = feaex.time_win(delta_ip, time1, win_size=0.030, step=0.001)
    X_ip = feaex.mean_win_data(X_ip)
    data_matrix = np.zeros((len(X_ip), col_num))

    # 降采样：对需要降采样的信号
    i = 0
    for channel in channels_down:
        time, data = fr.read_data(shot, channel, start, end)
        data_w, time, fs6 = feaex.time_win(data, time, win_size=0.030, step=0.001)
        data_wm = feaex.mean_win_data(data_w)
        data_matrix[:, i] = data_wm
        i += 1

    # 不需要降采样的信号直接读取
    for channel in channels:
        time, data = fr.read_data(shot, channel, start, end)
        data, time, fs6 = feaex.time_win(data, time, win_size=0.030, step=0.001)
        data = feaex.mean_win_data(data)
        data_matrix[:, i] = data
        i += 1

    return data_matrix


def set_build(shotset, channels_down, channels, errorshot, col_num):
    train_data_dis = np.empty([0, col_num])  # 空训练集
    for i in range(shotset.shape[0]):  # 破裂炮
        shot = shotset[i, 0]
        endtime = shotset[i, 2]
        try:
            # matrix1 = shot_data(shot, shotset[i, 1], channels_down, channels, endtime, col_num)
            # matrix1 = shot_data_win(shot, shotset[i, 1], channels_down, channels, endtime, col_num)  # 时间窗数据
            # matrix1 = shot_data_win_add(shot, shotset[i, 1], channels_down, channels, endtime, col_num)  # 时间窗数据 补充
            matrix1 = shot_data_fft(shot, shotset[i, 1], channels_down, channels, endtime, col_num)  # 点数据+fft
            train_data_dis = np.append(train_data_dis, matrix1, axis=0)
        except Exception as err:
            print(err)
            print('errorshot:{}'.format(shot))
            errorshot.append(shot)
    return train_data_dis, errorshot


if __name__ == '__main__':
    # train_dis_shot = np.load(r'dataset\train_dis_shot.npy')
    # test_dis_shot = np.load(r'dataset\test_dis_shot.npy')
    # train_undis_shot = np.load(r'dataset\train_undis_shot.npy')
    # test_undis_shot = np.load(r'dataset\test_undis_shot.npy')
    H_beta_shot = np.load(r'LHdataset\t2_H_beta.npy')
    # L_beta_train_shot = np.load(r'LHdataset\L_beta_train.npy')
    # L_beta_val_shot = np.load(r'LHdataset\L_beta_val.npy')
    # L_beta_test_shot = np.load(r'LHdataset\L_beta_test.npy')
    H_beta_train_shot = np.load(r'LHdataset\t2_L_beta_train.npy')
    H_beta_val_shot = np.load(r'LHdataset\t2_L_beta_val.npy')
    H_beta_test_shot = np.load(r'LHdataset\t2_L_beta_test.npy')
    error_d_shot = np.load(r'LHdataset\t2_L&H_win_errorshot.npy')
    H_beta_shot = error_del(H_beta_shot, error_d_shot)
    H_beta_train_shot = error_del(H_beta_train_shot, error_d_shot)
    H_beta_val_shot = error_del(H_beta_val_shot, error_d_shot)
    H_beta_test_shot = error_del(H_beta_test_shot, error_d_shot)

    channels = ["EFIT_BETA_T", "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
                "BT", "DENSITY", "W_E", "FIR01", 'DH', 'DV', "FIR03"]  # 不需要降采样信号
    channels_down = ["I_HA_N", "V_LOOP", "BOLD09", "BOLU10", "SX10", "BOLD03", "BOLD06", "BOLU03",
                     "BOLU06", "SX03", "SX06"]  # 需要降采样的信号
    fre_name = ['mpf', 'fmax', 'fmin', 'Ptotal', 'mean1', 'max1', 'min1', 'var1', 'mean2', 'max2', 'min2', 'var2',
                'mean3', 'max3', 'min3', 'var3', 'FT_abs', 'FT_freq']
    strmp04 = ['mp04_' + j for j in fre_name]
    strmp13 = ['mp13_' + j for j in fre_name]
    strnp04 = ['np04_' + j for j in fre_name]
    strnp09 = ['np09_' + j for j in fre_name]
    columns = ['deltaip', 'betaN', "I_HA_N", "V_LOOP", "BOLD09", "BOLU10", "SX10", "BOLD03", "BOLD06",
               "BOLU03", "BOLU06", "SX03", "SX06", "EFIT_BETA_T",
               "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
               "BT", "DENSITY04", "W_E", "DENSITY01", 'DH', 'DV', "DENSITY03", 'skewsx',
               'kurtsx', 'varsx', 'skewbolu', 'kurtbolu', 'varbolu', 'skewbold',
               'kurtbold', 'varbold'] + strmp04 + strmp13 + strnp04 + strnp09 + ['disrup_tag', '#', 'time', 'endtime']

    disr_tag = 1
    errorshot = []
    col_num = 111  # 包括了特征数、破裂标签、炮号、时间与结束时间的列数

    # matrix1 = shot_data_win(36171, disr_tag, channels_down, channels, 1886, col_num)  # test function

    # 训练集
    print('训练集数据处理...')
    train_data, errorshot = set_build(H_beta_train_shot, channels_down, channels, errorshot, col_num)  # 非破裂
    topdata_train = pd.DataFrame(train_data,
                                 columns=columns)  # 训练集

    # 验证集
    print('验证集数据处理...')
    val_data, errorshot = set_build(H_beta_val_shot, channels_down, channels, errorshot, col_num)
    topdata_val = pd.DataFrame(val_data,
                               columns=columns)  # 验证集

    print('测试集数据处理...')
    test_data, errorshot = set_build(H_beta_test_shot, channels_down, channels, errorshot, col_num)  # 非破裂
    topdata_test = pd.DataFrame(test_data,
                                columns=columns)  # 训练集
    # # 高β集
    print('高β集数据处理...')
    H_beta_data, errorshot = set_build(H_beta_shot, channels_down, channels, errorshot, col_num)
    topdata_H_beta = pd.DataFrame(H_beta_data,
                                  columns=columns)  # 验证集

    # scaler = StandardScaler().fit(toptrain_data[:, :16])  # 归一化
    # normalized_data = scaler.transform(toptrain_data[:, :16])
    # toptrain_data[:, :16] = normalized_data

    # topdata_train.to_csv('dataset/topdata_train.csv')
    # topdata_test.to_csv('dataset/topdata_test.csv')

    # topdata_test.to_csv('LHdataset/topdata_test.csv')
    # topdata_train.to_csv('LHdataset/topdata_train.csv')
    # topdata_val.to_csv('LHdataset/topdata_val.csv')
    # topdata_H_beta.to_csv('LHdataset/topdata_H_beta.csv')
    #
    # topdata_test.to_csv('LHdataset/topdata_H_test.csv')
    # topdata_train.to_csv('LHdataset/topdata_H_train.csv')
    # topdata_val.to_csv('LHdataset/topdata_H_val.csv')
