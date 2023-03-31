# @Time : 2022/2/18 15:39 
# @Author : zhongyu 
# @File : LightGBM_trainning.py

import numpy as np
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tuning_shot import DatasetBuild
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    print('Loading data...')
    # # load or create your dataset
    # df_train = pd.read_csv('LHdataset/topdata_train_reserve.csv', index_col=0)
    # df_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    # validset_b = np.load(r'LHdataset\L_beta_val.npy')
    #
    # # 混合L和H数据集
    # df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    # weight_1 = np.ones([df_train.shape[0]])
    # df_H_beta_mix = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    # weight_2 = np.ones([df_H_beta_mix.shape[0]]) * 1.5
    # w_train = list(np.append(weight_1, weight_2, axis=0))
    # df_L_val = pd.read_csv('LHdataset/topdata_val.csv', index_col=0)
    # df_H_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    # validset_b = np.load(r'LHdataset\L_beta_val.npy')
    # validset_a = np.load(r'LHdataset\H_beta_val.npy')
    # # df_validation = pd.read_csv('dataset/topdata_train.csv', index_col=0)
    # validset_b = np.append(validset_b, validset_a, axis=0)
    # newdf_H = pd.DataFrame(np.repeat(df_H_beta_mix.values, 6, axis=0))  # 加倍10炮数据，调节model-mix效果
    # newdf_H.columns = df_H_beta_mix.columns
    # # df_train = df_train.append(newdf_H, ignore_index=True)
    # df_train = df_train.append(df_H_beta_mix, ignore_index=True)
    # df_val = df_L_val.append(df_H_val, ignore_index=True)
    # df_val = df_H_val
    #
    # # # 10炮训练集
    # # df_train = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    # # df_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    # # validset_b = np.load(r'LHdataset\H_beta_val.npy')
    #
    # y_train = df_train['disrup_tag']
    # X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime', "BOLD03", "BOLD06",
    #                      "BOLU03", "BOLU06"], axis=1).values
    # y_val = df_val['disrup_tag']
    # X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime', "BOLD03", "BOLD06",
    #                      "BOLU03", "BOLU06"], axis=1).values
    # train_data, val_data, train_y, val_y = \
    #     train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=y_train)
    #
    # # create dataset for lightgbm
    # lgb_train = lgb.Dataset(X_train, y_train,
    #                         feature_name=['delta_ip', 'beta_N', "HA", "V_LOOP", "SX03", "SX06", "EFIT_BETA_T",
    #                                       "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
    #                                       "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"])
    # lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 使用类方法构造Lightgbm输入数据集
    cloumn = ['delta_ip', 'beta_N', "HA", "V_LOOP", "SX03", "SX06", "EFIT_BETA_T",
              "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
              "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"]
    path_train = 'LHdataset/topdata_train.csv'
    path_val = 'LHdataset/topdata_val.csv'
    path_val_info = r'LHdataset\L_beta_val.npy'
    path_mix = 'LHdataset/topdata_H_beta_mix.csv'
    path_h_mix = 'LHdataset/t1_topdata_H_val.csv'
    h_weight = DatasetBuild(path_train, path_val, path_val_info, cloumn)
    lgb_train, lgb_eval, X_val, y_val = h_weight.shap_weight(2.2, path_mix, path_h_mix)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'max_depth': 9,
        'num_leaves': 70,
        'learning_rate': 0.1,
        'feature_fraction': 0.86,
        'bagging_fraction': 0.73,
        'bagging_freq': 0,
        'verbose': 0,
        'cat_smooth': 10,
        'max_bin': 255,
        'min_data_in_leaf': 165,
        'lambda_l1': 0.03,
        'lambda_l2': 2.78,
        'is_unbalance': True,
        'min_split_gain': 0.3
    }

    evals_result = {}  # to record eval results for plotting

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets={lgb_train, lgb_eval},
                    evals_result=evals_result,
                    early_stopping_rounds=30)

    # print('Saving model...')
    # # save model to file
    # gbm.save_model('model/model_1.txt')
    #
    # print('Plotting feature importances...')
    # ax = lgb.plot_importance(gbm, max_num_features=10)
    # plt.show()

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_val, y_pred) ** 0.5)
    print('the roc is', metrics.roc_auc_score(y_val, y_pred))
