# @Time : 2022/5/15 13:38 
# @Author : zhongyu 
# @File : search_weight.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import test_analysis_shot as tas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
import tuning_shot as ts
import params_tuning as pt

if __name__ == '__main__':
    print('Loading data...')

    # 混合L和H数据集
    df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    df_H_beta_mix = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    weight_1 = np.ones([df_train.shape[0]])

    df_L_val = pd.read_csv('LHdataset/topdata_val.csv', index_col=0)
    df_H_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    validset_b = np.load(r'LHdataset\L_beta_val.npy')
    validset_a = np.load(r'LHdataset\H_beta_val.npy')
    # df_validation = pd.read_csv('dataset/topdata_train.csv', index_col=0)
    validset_b = np.append(validset_b, validset_a, axis=0)
    df_train = df_train.append(df_H_beta_mix, ignore_index=True)
    df_val = df_L_val.append(df_H_val, ignore_index=True)

    y_train = df_train['disrup_tag']
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    y_val = df_val['disrup_tag']
    X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    train_data, val_data, train_y, val_y = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=y_train)

    # create dataset for lightgbm
    weight_range = [2.2]
    max_auc = float('0')
    best_params = {}
    for k in weight_range:
        weight_2 = np.ones([df_H_beta_mix.shape[0]]) * k
        w_train = list(np.append(weight_1, weight_2, axis=0))
        lgb_train = lgb.Dataset(X_train, y_train, weight= w_train,
                                feature_name=['delta_ip', 'beta_N', "HA", "V_LOOP", "BOLD03", "BOLD06",
                                              "BOLU03", "BOLU06", "SX03", "SX06", "EFIT_BETA_T",
                                              "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
                                              "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"])
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'is_unbalance': True
        }
        params = ts.tuning_ac(params, lgb_train, lgb_eval, X_val, y_val)
        evals_result = {}  # to record eval results for plotting
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=300,
                        valid_sets=[lgb_train, lgb_eval],
                        evals_result=evals_result,
                        verbose_eval=100,
                        early_stopping_rounds=30)
        # lgb.plot_metric(booster=evals_result, metric='auc')
        y_pre = gbm.predict(X_val)
        print("auc:", metrics.roc_auc_score(y_val, y_pre))
        mean_auc = metrics.roc_auc_score(y_val, y_pre)
        if mean_auc >= max_auc:
            max_auc = mean_auc
            best_params['num_leaves'] = params['num_leaves']
            best_params['max_depth'] = params['max_depth']
            best_params['k'] = k
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']
    print('best params:',best_params)