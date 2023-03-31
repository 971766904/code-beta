# @Time : 2022/9/2 20:34 
# @Author : zhongyu 
# @File : drop_sample_by_shap.py
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # shap分析
    df_validation = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)

    # dsp = lgb.Booster(model_file='model/model_1.txt')
    # dsp = lgb.Booster(model_file='model/model_1_10_40_300.txt')
    dsp = lgb.Booster(model_file='model/model_L_5_20.txt')
    test_data = df_validation.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1)
    y = df_validation['disrup_tag']
    dsp.params["objective"] = "binary"
    explainer = shap.TreeExplainer(dsp)
    shap_values = explainer.shap_values(test_data)
    target = shap_values[1]

    reserve_v = 1.5
    sample_reserve = np.where((target[:, 4] < reserve_v) & (target[:, 4] > -reserve_v) & (target[:, 5] > -reserve_v)
                              & (target[:, 5] < reserve_v) & (target[:, 6] > -reserve_v) & (target[:, 6] < reserve_v)
                              & (target[:, 7] > -reserve_v) & (target[:, 7] < reserve_v))
    data_reserve = df_validation.iloc[sample_reserve[0], :]
    weight_1 = np.ones([df_validation.shape[0]])*0.1
    weight_1[sample_reserve[0]] =1
    # np.savetxt("LHdataset/weight1.csv",weight_1,delimiter=',',newline=',')
    # data_reserve.to_csv('LHdataset/topdata_train_reserve.csv')
