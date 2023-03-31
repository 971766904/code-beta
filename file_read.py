# @Time : 2021/12/17 16:13
# @Author : zhongyu
# @File : shot-choose.py

import numpy as np
import xlrd
import hdf5Reader2A as h5r
import matplotlib.pyplot as plt


def read_data(shot, channel, time_start, time_end):
    """

    :param shot: 炮号
    :param channel: 诊断tag
    :param time_start: 开始时间
    :param time_end: 结束时间
    :return:
    """
    time, data_beta = h5r.read_channel(shot, channel=channel, device="2a")
    time = np.round(time, 5)
    start = np.where(time > time_start)[0][0]
    end = np.where(time >= time_end)[0][0]
    data_beta = data_beta[start:end]
    time_beta = time[start:end]
    return time_beta, data_beta


def beta_max(shot, time_range):
    '''

    :param shot:  炮号
    :param time_range:  时间范围[开始，结束]
    :return: βt， Ip/r*Bt， βNmax
    '''
    time_beta_t, data_beta_t = read_data(shot, "EFIT_BETA_T", time_range[0], time_range[1])
    time_bt, data_bt = read_data(shot, "BT", time_range[0], time_range[1])
    time_r, data_r = read_data(shot, "EFIT_MINOR_R", time_range[0], time_range[1])
    time_ip, data_ip = read_data(shot, "IP", time_range[0], time_range[1])
    beta_N = data_beta_t / (data_ip / (1000 * data_r * data_bt * 0.0622))
    N_max = np.where(beta_N == np.max(beta_N))[0][0]
    return data_beta_t[N_max], data_ip[N_max] / (1000 * data_r[N_max] * data_bt[N_max] * 0.0622), np.max(beta_N)


if __name__ == '__main__':
    excel_path = r"H:\project-dp\dptime\disruptions1.xlsx"  #
    dp_book = xlrd.open_workbook(excel_path, encoding_override="utf-8")
    dp_sheet = dp_book.sheet_by_index(0)

    dis_beta = []
    dis_tor = []
    dis_maxn = []
    undis_beta = []
    undis_tor = []
    undis_maxn = []
    shotset = []
    nrowsum = dp_sheet.nrows
    for i in range(0, nrowsum):
        if h5r.if_channel_exist(dp_sheet.row_values(i)[0], channel="EFIT_LI", device="2a") and h5r.if_channel_exist(
                dp_sheet.row_values(i)[0], channel="BT", device="2a"):
            time, data_beta = h5r.read_channel(dp_sheet.row_values(i)[0], channel="EFIT_LI", device="2a")
            if dp_sheet.row_values(i)[2] / 1000 < np.round(time, 5)[-1]:
                shotset.append(dp_sheet.row_values(i)[0])
                if dp_sheet.row_values(i)[1]:
                    dis_b, dis_t, dis_max = beta_max(dp_sheet.row_values(i)[0], [0.07, dp_sheet.row_values(i)[2] / 1000])
                    dis_beta.append(dis_b)
                    dis_tor.append(dis_t)
                    dis_maxn.append(dis_max)
                else:
                    undis_b, undis_t, undis_max = beta_max(dp_sheet.row_values(i)[0], [0.07, dp_sheet.row_values(i)[2] / 1000])
                    undis_beta.append(undis_b)
                    undis_tor.append(undis_t)
                    undis_maxn.append(undis_max)
        else:
            pass
    # beta_n_max &ip*a*bt

    plt.figure()
    x1 = np.linspace(0, 1, 50)
    y1 = 3.6 * x1
    y2 = 0.155 * x1
    plt.scatter(dis_tor, dis_beta, color='r')
    plt.scatter(undis_tor, undis_beta, color='b')
    plt.ylabel('$β_t$[%]')
    plt.xlabel('$I_p$/($a_p$$B_t$)[MA/mT]')
    plt.ylim((0, 2))
    plt.plot(x1, y1, dashes=[6, 2])  # 虚线βN
    plt.text(0.29, 1.15, '$β_N$=3.6')
    plt.text(0.48, 0.09, '$β_N$=0.15')
    plt.plot(x1, y2, dashes=[6, 2])  # 虚线βN
    plt.show()

    # time_li, data_li = read_data(35944, "EFIT_li", time_range[0], time_range[1])
    time_ip, data_ip = read_data(36627, "IP", 0.07, 1.153)
    plt.figure()
    plt.plot(time_ip, data_ip, color='r')
    plt.show()