# @Time : 2021/12/17 16:13 
# @Author : zhongyu
# @File : shot-choose.py
import numpy as np
import xlrd
import hdf5Reader2A as h5r
from sklearn.model_selection import train_test_split
import file_read as fr

if __name__ == '__main__':
    # H_beta_shot = np.load(r'LHdataset\H_beta1.npy')
    excel_path = r"H:\project-dp\dptime\disruptions1.xlsx"  #
    dp_book = xlrd.open_workbook(excel_path, encoding_override="utf-8")
    dp_sheet = dp_book.sheet_by_index(0)

    device = '2A'
    channels = ["EFIT_BETA_T", "IP", "BT", "FIR01","FIR04","SX03", "BOLU03", "BOLD03", "W_E",
                "V_LOOP", "DENSITY", "IP_TARGET", "I_HA_N",'DH', 'DV']  # 只用一个efit，减少复杂度

    # 信号是否都存在
    signalfail_shot = []  # 信号缺失炮或非自然破裂炮
    validshot = np.empty([0, 3])  # 空集
    nrowsum = dp_sheet.nrows
    for i in range(0, nrowsum):
        fail_mark = 0
        shot = dp_sheet.row_values(i)[0]
        print(shot)
        for channel in channels:
            cha_out = h5r.if_channel_exist(shot, channel, device)
            if not cha_out:
                signalfail_shot.append(shot)
                print(channel)
                fail_mark = 1
                break
            else:
                time, data_beta = h5r.read_channel(shot, channel="EFIT_LI", device="2a")
                start = h5r.get_attrs("StartTime", shot_number=shot, channel="EFIT_LI")
                end = dp_sheet.row_values(i)[2] / 1000
                if (end - start) < 0.15:  # 放电平台时间太短
                    signalfail_shot.append(shot)
                    print('flattop time not enough')
                    fail_mark = 1
                if dp_sheet.row_values(i)[2] / 1000 > np.round(time, 5)[-1]:  # 排除EFIT数据在endtime前有缺失的炮
                    signalfail_shot.append(shot)
                    print('EFIT fail')
                    fail_mark = 1
                    break
        if not fail_mark:
            validshot = np.append(validshot, [dp_sheet.row_values(i)], axis=0)

    # 高β和低β划分
    H_beta = np.empty([0, 3])
    L_beta = np.empty([0, 3])
    for i in range(validshot.shape[0]):
        start = h5r.get_attrs("StartTime", shot_number=validshot[i, 0], channel="EFIT_LI")
        end = validshot[i, 2] / 1000
        dis_b, dis_t, dis_max = fr.beta_max(validshot[i, 0], [start, end])
        if validshot[i, 1]:
            validshot[i, 1] = 1
        if dis_max <= 1:
            L_beta = np.append(L_beta, [validshot[i, :]], axis=0)
        else:
            H_beta = np.append(H_beta, [validshot[i, :]], axis=0)
    # 删除用于混合的10炮
    some_values = [36389, 36590, 36712, 36720, 36827, 36940, 36972, 37009,
                   37030, 37042, 36992, 36892, 36795, 37080, 36790, 36804, 36680, 37033, 36944]
    r_del = []
    for j in range(H_beta.shape[0]):
        if H_beta[j, 0] in some_values:
            r_del.append(j)
    H_beta = np.delete(H_beta, r_del, axis=0)



    # sklearn 划分数据集
    train1_data, test_data, train1_y, test_y = \
        train_test_split(H_beta, H_beta[:, 1], test_size=0.2, random_state=1, shuffle=True, stratify=H_beta[:, 1])
    train_data, val_data, train_y, val_y = \
        train_test_split(train1_data, train1_y, test_size=0.2, random_state=1, shuffle=True, stratify=train1_y)

    # # 保存
    # np.save(r'dataset\train_dis_shot.npy', train_dis_shot)
    # np.save(r'dataset\test_dis_shot.npy', test_dis_shot)
    # np.save(r'dataset\train_undis_shot.npy', train_undis_shot)
    # np.save(r'dataset\test_undis_shot.npy', test_undis_shot )

    # np.save(r'LHdataset\H_beta.npy', H_beta)
    # np.save(r'LHdataset\L_beta_train.npy', train_data)
    # np.save(r'LHdataset\L_beta_val.npy', val_data)
    # np.save(r'LHdataset\L_beta_test.npy', test_data)
    #
    # np.save(r'LHdataset\H_beta_train.npy', train_data)
    # np.save(r'LHdataset\H_beta_val.npy', val_data)
    # np.save(r'LHdataset\H_beta_test.npy', test_data)
