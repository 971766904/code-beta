# @Time : 2022/4/18 16:48 
# @Author : zhongyu 
# @File : shot_signal_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import hdf5Reader2A as h5r
import file_read as fr
import matplotlib.font_manager as fm

if __name__ == '__main__':
    H_beta_shot = np.load(r'LHdataset\H_beta_test.npy')
    shot_for_shap = [36755, 36823, 37021]
    # shot_for_shap = [36755]
    feature_tags = ["DENSITY", "W_E", "V_LOOP", "FIR01", "FIR03", "BOLD03", "BOLD06", "BOLU03", "BOLU06"]

    for k in shot_for_shap:
        va_shot = k
        # time, data_ip = h5r.read_channel(va_shot, channel='IP', device="2a")

        fig = plt.figure()

        # 微软雅黑,如果需要宋体,可以用simsun.ttc
        myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
        fronts = {'family': 'Times New Roman', 'size': 12}
        i = 1
        for tag in feature_tags:
            time, data = h5r.read_channel(va_shot, channel=tag, device="2a")
            fig.add_subplot(3, 3, i)
            i += 1
            plt.title('#{}'.format(va_shot)+'-{}'.format(tag))
            plt.plot(time, data, label=tag)
            plt.xlabel('time', fontproperties=myfont)
            plt.ylabel(tag, fontproperties=myfont)
            plt.xticks(fontproperties='Times New Roman', fontsize=12)
            plt.yticks(fontproperties='Times New Roman', fontsize=12)
        plt.legend()
