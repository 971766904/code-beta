# @Time : 2022/5/10 9:13 
# @Author : zhongyu 
# @File : test for shap plot.py
import xgboost
import shap
import lightgbm as lgb
import pandas as pd

# train XGBoost model
df_validation = pd.read_csv('LHdataset/topdata_test.csv', index_col=0)

# dsp = lgb.Booster(model_file='model/model_1.txt')
# dsp = lgb.Booster(model_file='model/model_1_10_40_300.txt')
dsp = lgb.Booster(model_file='model/model_L_5_20.txt')
columns = ['Δip', 'βN', "I_HA_N", "V_LOOP", "BOLD03", "BOLD06", "BOLU03",
           "BOLU06", "SX03", "SX06", "EFIT_BETA_T", "EFIT_BETA_P",
           "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
           "DENSITY", "W_E", "FIR01", "FIR03"]

test_data = df_validation.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
y = df_validation['disrup_tag']
dsp.params["objective"] = "binary"
dsp = lgb.LGBMClassifier().fit(test_data,y)

# compute SHAP values
explainer = shap.Explainer(dsp, test_data)
shap_values = explainer(test_data,check_additivity=False)

shap.plots.bar(shap_values)