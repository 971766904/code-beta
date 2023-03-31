# @Time : 2022/2/19 16:37 
# @Author : zhongyu 
# @File : tuning_shot.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import test_analysis_shot as tas
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import coral_application as capp
from tsfresh import extract_features, extract_relevant_features, select_features
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

def grid_search(X,y,params_test1):
    kf = KFold(n_splits=2, shuffle=True, random_state=123)  # 后面KFold网格搜索和交叉验证用

    gbm = lgb.LGBMClassifier(objective='binary',
                             is_unbalance=True,
                             metric='auc',
                             num_leaves=40,
                             learning_rate=0.1,
                             n_estimators=400, bagging_fraction = 0.8,feature_fraction = 0.8
                             )
    gsearch1 = GridSearchCV(estimator=gbm, param_grid=params_test1, scoring='roc_auc', cv=5,
                            verbose=1)

    gsearch1.fit(X, y)
    return gsearch1.best_params_, gsearch1.best_score_



def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


def train_assess(params, lgb_train, lgb_eval, validset_b, df_validation, a1, delta_t):
    evals_result = {}  # to record eval results for plotting

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets=[lgb_train, lgb_eval],
                    evals_result=evals_result,
                    verbose_eval=20,
                    early_stopping_rounds=30)
    predict_result = tas.assess1(validset_b, df_validation, a1, delta_t, gbm)
    return predict_result


def tuning_ac(params, lgb_train, lgb_eval, X_val, y_val):
    max_auc = float('0')
    best_params = {}

    # 准确率
    print("调参1：提高准确率")
    for max_depth in [3, 4, 5, 6]:
        for num_leaves in [50, 55, 40, 45, 30, 35, 60, 65, 70]:
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth

            # cv_results = lgb.cv(
            #     params,
            #     lgb_train,
            #     seed=1,
            #     nfold=5,
            #     metrics=['auc'],
            #     early_stopping_rounds=10,
            #     verbose_eval=50
            # )
            #
            # mean_auc = pd.Series(cv_results['auc-mean']).max()
            # boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
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
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    print('max auc', max_auc, 'best params', best_params)

    return params


def trans_a(path, data):
    A = np.load(path)
    Z = np.dot(A.T, data.T)
    Z /= np.linalg.norm(Z, axis=0)
    return Z.T


def trans_ma(A, data):
    Z = np.dot(A.T, data.T)
    Z /= np.linalg.norm(Z, axis=0)
    return Z.T


class DatasetBuild:
    def __init__(self, train_path, val_path, val_infor_path, col_name):
        self.train_path = train_path
        self.val_path = val_path
        self.val_infor_path = val_infor_path
        self.col_name = col_name

    def normal(self):
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_val = pd.read_csv(self.val_path, index_col=0)
        y_train = df_train['disrup_tag']
        X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1)
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1)
        # X_train = trans_a(r'LHdataset\JDA_A_maxacc.npy', X_train)
        # X_val = trans_a(r'LHdataset\JDA_A_maxacc.npy', X_val)

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train, feature_name=self.col_name
                                )
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        return lgb_train, lgb_eval, X_val, y_val

    def weightmix(self, weight, mix_path, h_path):
        # 混合L和H数据集
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_H_beta_mix = pd.read_csv(mix_path, index_col=0)
        weight_1 = np.ones([df_train.shape[0]])

        df_L_val = pd.read_csv(self.val_path, index_col=0)
        df_H_val = pd.read_csv(h_path, index_col=0)
        df_train = df_train.append(df_H_beta_mix, ignore_index=True)
        df_val = df_H_val

        y_train = df_train['disrup_tag']
        X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        X_train = trans_a(r'LHdataset\JDA_A_maxacc.npy', X_train)
        X_val = trans_a(r'LHdataset\JDA_A_maxacc.npy', X_val)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X_train, y_train)
        Y_tar_pseudo = clf.predict(X_val)
        acc = sklearn.metrics.accuracy_score(y_val, Y_tar_pseudo)

        print('JDA iteration : Acc: {:.4f}'.format(acc))

        # create dataset for lightgbm
        weight_2 = np.ones([df_H_beta_mix.shape[0]]) * weight
        w_train = list(np.append(weight_1, weight_2, axis=0))
        lgb_train = lgb.Dataset(X_train, y_train, weight=w_train,
                                feature_name=self.col_name)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)


    def shap_weight(self,weight, mix_path, h_path):
        # 混合L和H数据集
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_H_beta_mix = pd.read_csv(mix_path, index_col=0)
        weight_1 = np.load('LHdataset/weight1.npy')


        df_H_val = pd.read_csv(h_path, index_col=0)
        df_train = df_train.append(df_H_beta_mix, ignore_index=True)
        df_val = df_H_val

        y_train = df_train['disrup_tag']
        X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values


        # create dataset for lightgbm
        weight_2 = np.ones([df_H_beta_mix.shape[0]]) * weight
        w_train = list(np.append(weight_1, weight_2, axis=0))
        lgb_train = lgb.Dataset(X_train, y_train, weight=w_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        return lgb_train,lgb_eval,X_val,y_val



    def shotmix(self, mixnum: int, mix_path, h_path, some_values):
        # 混合L和H数据集
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_H_beta_mix = pd.read_csv(mix_path, index_col=0)
        df_L_val = pd.read_csv(self.val_path, index_col=0)
        df_H_val = pd.read_csv(h_path, index_col=0)
        # some_values = [36389, 36590, 36712, 36720, 36827, 36940, 36972, 37009,
        #                37030, 37042]

        print('shot number', mixnum)
        mix_shot = df_H_beta_mix.loc[df_H_beta_mix['#'].isin(some_values[:mixnum])]
        df_train = df_train.append(mix_shot, ignore_index=True)
        df_val = df_L_val.append(df_H_val, ignore_index=True)

        y_train = df_train['disrup_tag']
        X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train,
                                feature_name=self.col_name)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        return lgb_train, lgb_eval, X_val, y_val

    def disol(self, tra_infor_path):
        trainset = np.load(tra_infor_path)
        disshot = trainset[np.where(trainset == 1)[0], 0]
        undisshot = trainset[np.where(trainset == 0)[0], 0]
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_val = pd.read_csv(self.val_path, index_col=0)
        disshotdata = df_train.loc[df_train['#'].isin(disshot)]
        undisdata = df_train.loc[df_train['#'].isin(undisshot)]
        disdata = disshotdata.loc[disshotdata['disrup_tag'] == 1]
        df_f_train = undisdata.append(disdata, ignore_index=True)

        y_train = df_f_train['disrup_tag']
        X_train = df_f_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train,
                                feature_name=self.col_name)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        return lgb_train, lgb_eval, X_val, y_val

    def label_balan(self, tra_infor_path):
        trainset = np.load(tra_infor_path)
        disshot = trainset[np.where(trainset == 1)[0], 0]
        undisshot = trainset[np.where(trainset == 0)[0], 0]
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_val = pd.read_csv(self.val_path, index_col=0)
        disshotdata = df_train.loc[df_train['#'].isin(disshot)]
        undisdata = df_train.loc[df_train['#'].isin(undisshot)]
        disdata = disshotdata.loc[disshotdata['disrup_tag'] == 1]
        df_f_train = undisdata.append(disdata, ignore_index=True)
        disval = df_val.loc[df_val['disrup_tag'] == 1]
        undisval = df_val.loc[df_val['disrup_tag'] == 0]
        undisval_ba = undisval.sample(disval.shape[0], random_state=1)
        df_val = disval.append(undisval_ba, ignore_index=True)

        y_train = df_f_train['disrup_tag']
        X_train = df_f_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train,
                                feature_name=list(
                                    df_f_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).columns))
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        return lgb_train, lgb_eval, X_val, y_val,X_train,y_train

    def add_data_join(self, train_add, val_add, tra_infor_path):
        trainset = np.load(tra_infor_path)
        disshot = trainset[np.where(trainset == 1)[0], 0]
        undisshot = trainset[np.where(trainset == 0)[0], 0]
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_train_add = pd.read_csv(train_add, index_col=0)
        df_val = pd.read_csv(self.val_path, index_col=0)
        df_val_add = pd.read_csv(val_add, index_col=0)
        df_train = df_train.join(df_train_add)
        df_val = df_val.join(df_val_add)
        disshotdata = df_train.loc[df_train['#'].isin(disshot)]
        undisdata = df_train.loc[df_train['#'].isin(undisshot)]
        disdata = disshotdata.loc[disshotdata['disrup_tag'] == 1]
        df_f_train = undisdata.append(disdata, ignore_index=True)
        disval = df_val.loc[df_val['disrup_tag'] == 1]
        undisval = df_val.loc[df_val['disrup_tag'] == 0]
        undisval_ba = undisval.sample(disval.shape[0], random_state=1)
        df_val = disval.append(undisval_ba, ignore_index=True)

        y_train = df_f_train['disrup_tag']
        X_train = df_f_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train,
                                feature_name=list(
                                    df_f_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).columns))
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        return lgb_train, lgb_eval, X_val, y_val, X_train,y_train

    def coral_conv(self, mix_path):
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_val = pd.read_csv(self.val_path, index_col=0)
        df_H_beta_mix = pd.read_csv(mix_path, index_col=0)
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1)
        X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        X_t = df_H_beta_mix.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y_train = df_train['disrup_tag'].values
        y_t = df_H_beta_mix['disrup_tag'].values
        X_train = capp.batch_cov(X_train, X_t)  # CORAL方法
        # X_train = jda_con(X_train, y_train, X_t, y_t)  # JDA方法
        Xs_new = np.append(X_train, X_t, axis=0)
        ys_new = np.append(y_train, y_t, axis=0)

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(Xs_new, ys_new, feature_name=['delta_ip', 'beta_N', "HA", "V_LOOP", "BOLD03", "BOLD06",
                                                              "BOLU03", "BOLU06", "SX03", "SX06", "EFIT_BETA_T",
                                                              "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
                                                              "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"])
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        return lgb_train, lgb_eval, X_val, y_val

    def drop_feature(self, fea_name):
        df_train = pd.read_csv(self.train_path, index_col=0)
        df_val = pd.read_csv(self.val_path, index_col=0)
        y_train = df_train['disrup_tag']
        X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'] + fea_name, axis=1)
        y_val = df_val['disrup_tag']
        X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'] + fea_name, axis=1)

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train, feature_name=list(X_train.columns)
                                )
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        return lgb_train, lgb_eval, X_val, y_val


def rate_search(rate_dic, path_mix, path_h_mix):
    params_dic = dict()
    for rate in rate_dic.keys():
        lgb_train, lgb_eval, X_val, y_val = h_weight.shotmix(-1, path_mix, path_h_mix, rate_dic[rate])
        # 准确率
        print("调参1：提高准确率")
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'is_unbalance': True
        }

        max_auc = float('0')
        best_params = {}
        a1 = 0.8
        delta_t = 90
        auc_l = []

        # 准确率
        print("调参1：提高准确率")
        for max_depth in [3, 4, 5, 6]:
            for num_leaves in [50, 55, 65, 60, 25, 30, 45]:
                params['num_leaves'] = num_leaves
                params['max_depth'] = max_depth

                evals_result = {}  # to record eval results for plotting
                gbm = lgb.train(params,
                                lgb_train,
                                num_boost_round=300,
                                valid_sets=[lgb_train, lgb_eval],
                                evals_result=evals_result,
                                verbose_eval=100,
                                early_stopping_rounds=30)

                y_pre = gbm.predict(X_val)
                print("auc:", metrics.roc_auc_score(y_val, y_pre))
                mean_auc = metrics.roc_auc_score(y_val, y_pre)
                auc_l = np.append(auc_l, mean_auc)

                if mean_auc >= max_auc:
                    max_auc = mean_auc
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth
        if 'num_leaves' and 'max_depth' in best_params.keys():
            params['num_leaves'] = best_params['num_leaves']
            params['max_depth'] = best_params['max_depth']

        print(best_params)
        print(params)
        params_dic[rate] = params
    return params_dic


if __name__ == '__main__':
    print('Loading data...')
    # load or create your dataset
    path_train = 'LHdataset/t2_point_fft_topdata_train.csv'
    path_val = 'LHdataset/t2_point_fft_topdata_val.csv'
    path_val_info = r'LHdataset\t2_L_beta_val.npy'
    path_train_info = r'LHdataset\t2_L_beta_train.npy'
    path_mix = 'LHdataset/topdata_H_beta_mix.csv'
    path_h_mix = 'LHdataset/topdata_H_val.csv'
    path_train_add = 'LHdataset/t2_win_topdata_train_add.csv'
    path_val_add = 'LHdataset/t2_win_topdata_val_add.csv'
    fre_name = ['mpf', 'fmax', 'fmin', 'Ptotal', 'mean1', 'max1', 'min1', 'var1', 'mean2', 'max2', 'min2', 'var2',
                'mean3', 'max3', 'min3', 'var3', 'FT_abs', 'FT_freq']
    strmp04 = ['mp04_' + j for j in fre_name]
    strmp13 = ['mp13_' + j for j in fre_name]
    strnp04 = ['np04_' + j for j in fre_name]
    strnp09 = ['np09_' + j for j in fre_name]
    columns = ['deltaip', 'betaN', "I_HA_N", "V_LOOP", "BOLD09", "BOLU10", "SX10", "EFIT_BETA_T",
               "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
               "BT", "DENSITY04", "W_E", "DENSITY01", 'DH', 'DV', 'skewsx',
               'kurtsx', 'varsx', 'skewbolu', 'kurtbolu', 'varbolu', 'skewbold',
               'kurtbold', 'varbold'] + strmp04 + strmp13 + strnp04 + strnp09

    h_weight = DatasetBuild(path_train, path_val, path_val_info, columns)
    rate_dic = {'1': [36389, 36827, 36972, 37030, 36992, 36892, 36795, 37080, 36790, 36804],
                '0.8': [36389, 36827, 36590, 37030, 36992, 36892, 36795, 37080, 37042, 36804],
                '0.6': [36389, 36827, 36590, 37030, 36992, 36940, 36795, 37080, 37042, 36712],
                '0.4': [37009, 36827, 36590, 37030, 36992, 36940, 36795, 36944, 37042, 36712],
                '0.2': [37009, 36680, 36590, 37030, 36992, 36940, 37033, 36944, 37042, 36712]}
    params_dic = {'1': {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
                        'learning_rate': 0.1, 'is_unbalance': True,
                        'num_leaves': 30,
                        'max_depth': 6},
                  '0.8': {'boosting_type': 'gbdt',
                          'objective': 'binary',
                          'metric': 'auc',
                          'learning_rate': 0.1,
                          'is_unbalance': True,
                          'num_leaves': 25,
                          'max_depth': 6},
                  '0.6': {'boosting_type': 'gbdt',
                          'objective': 'binary',
                          'metric': 'auc',
                          'learning_rate': 0.1,
                          'is_unbalance': True,
                          'num_leaves': 25,
                          'max_depth': 6},
                  '0.4': {'boosting_type': 'gbdt',
                          'objective': 'binary',
                          'metric': 'auc',
                          'learning_rate': 0.1,
                          'is_unbalance': True,
                          'num_leaves': 25,
                          'max_depth': 6},
                  '0.2': {'boosting_type': 'gbdt',
                          'objective': 'binary',
                          'metric': 'auc',
                          'learning_rate': 0.1,
                          'is_unbalance': True,
                          'num_leaves': 30,
                          'max_depth': 6}}
    drop_name = ['mp04_fmax', 'mp04_mean3', 'np09_var3', 'mp13_var3', 'np04_mean3', 'mp04_mean2', 'np04_fmin',
                 'np04_var3']
    lgb_train, lgb_eval, X_val, y_val,X_train,y_trian = h_weight.label_balan(path_train_info)

    # 搜索删除部分相反信号的参数
    # 混合L和H数据集
    df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    weight_1 = np.ones([df_train.shape[0]])
    df_H_beta_mix = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    weight_2 = np.ones([df_H_beta_mix.shape[0]]) * 1.5
    w_train = list(np.append(weight_1, weight_2, axis=0))
    df_L_val = pd.read_csv('LHdataset/topdata_val.csv', index_col=0)
    df_H_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    validset_b = np.load(r'LHdataset\L_beta_val.npy')
    validset_a = np.load(r'LHdataset\H_beta_val.npy')
    # df_validation = pd.read_csv('dataset/topdata_train.csv', index_col=0)
    validset_b = np.append(validset_b, validset_a, axis=0)
    newdf_H = pd.DataFrame(np.repeat(df_H_beta_mix.values, 6, axis=0))  # 加倍10炮数据，调节model-mix效果
    newdf_H.columns = df_H_beta_mix.columns
    # df_train = df_train.append(newdf_H, ignore_index=True)
    df_train = df_train.append(df_H_beta_mix, ignore_index=True)
    df_val = df_L_val.append(df_H_val, ignore_index=True)
    df_val = df_H_val

    # # 10炮训练集
    # df_train = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    # df_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    # validset_b = np.load(r'LHdataset\H_beta_val.npy')

    y_train = df_train['disrup_tag']
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime', "BOLD03", "BOLD06",
                             "BOLU03", "BOLU06"], axis=1).values
    y_val = df_val['disrup_tag']
    X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime', "BOLD03", "BOLD06",
                         "BOLU03", "BOLU06"], axis=1).values
    train_data, val_data, train_y, val_y = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=y_train)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train,
                            feature_name=['delta_ip', 'beta_N', "HA", "V_LOOP", "SX03", "SX06", "EFIT_BETA_T",
                                          "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
                                          "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"])
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # lgb_train, lgb_eval, X_val, y_val = h_weight.disol(path_train_info)
    # X_filtered = select_features(X, y)

    # # 训练模型
    # params = {'boosting_type': 'goss', 'objective': 'binary', 'metric': {'auc'}, 'learning_rate': 0.03,
    #           'is_unbalance': True, 'boost_from_average': False, "top_rate": 0.3,'min_data_in_leaf': 40,
    #           "other_rate": 0.8 - 0.3, 'bagging_fraction': 0.8,'feature_fraction': 0.8,
    #           'num_leaves': 10}
    # evals_result = {}  # to record eval results for plotting
    #
    # print('Starting training...')
    # # train
    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 num_boost_round=300,
    #                 valid_sets={lgb_train, lgb_eval},
    #                 evals_result=evals_result,
    #                 verbose_eval=100,
    #                 early_stopping_rounds=30)
    # ax = lgb.plot_importance(gbm, max_num_features=10)
    #
    # vaild_preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    # results = []
    # for pred in vaild_preds:
    #     result = 1 if pred > 0.5 else 0
    #     results.append(result)
    # f1 = f1_score(results, y_val, average='macro')
    # params_test1 = {'num_leaves': range(50, 100, 5),
    #                 'min_data_in_leaf': range(20,200,20)}

    # best_params =grid_search(X_train,y_trian,params_test1)

    # print('Saving model...')
    # # save model to file
    # gbm.save_model('model/model_rate{}_mix.txt'.format(rate))

    # 准确率
    print("调参1：提高准确率")
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
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
    # 最佳迭代次数
    # cv_results = lgb.cv(params, lgb_train, num_boost_round=5000, nfold=5, stratified=False, shuffle=True, metrics='auc',
    #                     early_stopping_rounds=50, seed=0)
    # print('best n_estimators:', len(cv_results['auc-mean']))
    #
    # print('best cv score:', pd.Series(cv_results['auc-mean']).max())

    max_auc = float('0')
    best_params = {}
    a1 = 0.8
    delta_t = 90
    auc_l = []

    # 准确率
    print("调参1：提高准确率")
    for max_depth in [3, 4, 5, 6,7,8,9,10,11]:
        for num_leaves in [50, 55, 65, 60, 25, 30, 45, 20, 15, 10,60,70,80,90,100,110,120,130]:
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth

            # predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_val, a1, delta_t)
            # prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
            # mean_auc = prf_r[1]
            # auc_l = np.append(auc_l,mean_auc)
            evals_result = {}  # to record eval results for plotting
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=300,
                            valid_sets=[lgb_train, lgb_eval],
                            evals_result=evals_result,
                            verbose_eval=100,
                            early_stopping_rounds=30)

            y_pre = gbm.predict(X_val)
            print("auc:", metrics.roc_auc_score(y_val, y_pre))
            mean_auc = metrics.roc_auc_score(y_val, y_pre)
            auc_l = np.append(auc_l, mean_auc)

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']

    # 过拟合
    # print("调参2：降低过拟合")
    # for max_bin in [64,128,256,512]:
    #     for min_data_in_leaf in [18,19,20,21,22]:
    #         params['max_bin'] = max_bin
    #         params['min_data_in_leaf'] = min_data_in_leaf
    #
    #         predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    #         prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
    #         mean_auc = prf_r[1]
    #
    #         if mean_auc >= max_auc:
    #             max_auc = mean_auc
    #             best_params['max_bin'] = max_bin
    #             best_params['min_data_in_leaf'] = min_data_in_leaf
    # if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
    #     params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    #     params['max_bin'] = best_params['max_bin']
    #
    # print("调参3：降低过拟合")
    # for feature_fraction in [0.6, 0.8, 1]:
    #     for bagging_fraction in [0.8,0.9,1]:
    #         for bagging_freq in [2,3,4]:
    #             params['feature_fraction'] = feature_fraction
    #             params['bagging_fraction'] = bagging_fraction
    #             params['bagging_freq'] = bagging_freq
    #
    #             predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    #             prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
    #             mean_auc = prf_r[1]
    #
    #             if mean_auc >= max_auc:
    #                 max_auc = mean_auc
    #                 best_params['feature_fraction'] = feature_fraction
    #                 best_params['bagging_fraction'] = bagging_fraction
    #                 best_params['bagging_freq'] = bagging_freq
    #
    # if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
    #     params['feature_fraction'] = best_params['feature_fraction']
    #     params['bagging_fraction'] = best_params['bagging_fraction']
    #     params['bagging_freq'] = best_params['bagging_freq']
    #
    # print("调参4：降低过拟合")
    # for lambda_l1 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
    #     for lambda_l2 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
    #         params['lambda_l1'] = lambda_l1
    #         params['lambda_l2'] = lambda_l2
    #
    #         predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    #         prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
    #         mean_auc = prf_r[1]
    #
    #         if mean_auc >= max_auc:
    #             max_auc = mean_auc
    #             best_params['lambda_l1'] = lambda_l1
    #             best_params['lambda_l2'] = lambda_l2
    # if 'lambda_l1' and 'lambda_l2' in best_params.keys():
    #     params['lambda_l1'] = best_params['lambda_l1']
    #     params['lambda_l2'] = best_params['lambda_l2']
    #
    # print("调参5：降低过拟合2")
    # for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    #     params['min_split_gain'] = min_split_gain
    #     predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    #     prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
    #     mean_auc = prf_r[1]
    #
    #     if mean_auc >= max_auc:
    #         max_auc = mean_auc
    #
    #         best_params['min_split_gain'] = min_split_gain
    # if 'min_split_gain' in best_params.keys():
    #     params['min_split_gain'] = best_params['min_split_gain']

    print(best_params)
    print(params)
