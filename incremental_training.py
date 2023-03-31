# @Time : 2022/4/12 15:05 
# @Author : zhongyu 
# @File : incremental_training.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import test_analysis_shot as tas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

if __name__ == '__main__':
    print('Loading data...')
    # # load or create your dataset
    # 10炮训练集
    df_train = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    df_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    validset_b = np.load(r'LHdataset\H_beta_val.npy')
    newdf_H = pd.DataFrame(np.repeat(df_train.values, 24, axis=0))
    newdf_H.columns = df_train.columns
    df_train = df_train.append(newdf_H, ignore_index=True)

    # save_tag = input("save it or not?")

    y_train = df_train['disrup_tag']
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    y_val = df_val['disrup_tag']
    X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    train_data, val_data, train_y, val_y = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=y_train)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_data, train_y, feature_name=['delta_ip', 'beta_N', "HA", "V_LOOP", "BOLD03", "BOLD06",
                                                               "BOLU03", "BOLU06", "SX03", "SX06", "EFIT_BETA_T",
                                                               "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
                                                               "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"])
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # # bayes调超参
    # bht.optimize_lgb(X_train, y_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'max_depth': 5,
        'num_leaves': 20,
        'learning_rate': 0.1,
        'feature_fraction': 0.86,
        'bagging_fraction': 0.73,
        'min_data_in_leaf': 165,
        'lambda_l1': 0.03,
        'lambda_l2': 2.78,
        'is_unbalance': True
    }

    evals_result = {}  # to record eval results for plotting
    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    init_model='model/model_L_5_20.txt',
                    num_boost_round=300,
                    valid_sets=[lgb_train, lgb_eval],
                    evals_result=evals_result,
                    verbose_eval=100,
                    keep_training_booster=True,
                    early_stopping_rounds=30)
    lgb.plot_metric(booster=evals_result, metric='auc')
    # gbm.save_model('model/model_incremental_5_20.txt')