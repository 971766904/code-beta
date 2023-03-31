# @Time : 2022/5/16 16:57 
# @Author : zhongyu 
# @File : shap&signal_fig.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import hdf5Reader2A as h5r
import file_read as fr
import matplotlib.font_manager as fm
import lightgbm as lgb
import shap
import test_analysis_shot as tas
import pandas as pd
from file_read import read_data


class Plot2d:
    def __init__(self, array_t, shot):
        self.array_t = array_t
        self.shot = shot

    def array_plot(self):
        time, data = h5r.read_channel(self.shot, channel=self.array_t[0], device="2a")
        local = np.arange(len(self.array_t)) + 1
        array_1 = np.empty([len(self.array_t), data.shape[0]])
        for i in range(len(self.array_t)):
            time, data = h5r.read_channel(self.shot, channel=self.array_t[i], device="2a")
            array_1[i, :] = data
        plt.figure()
        plt.contourf(time, local, array_1)
        plt.colorbar()
        plt.xlabel('time(s)')


class Plotsignal:
    def __init__(self, tag_list, shot):
        self.tag_list = tag_list
        self.shot = shot

    def signal_plot(self):
        fig, axes = plt.subplots(nrows=int(len(self.tag_list) / 2), ncols=1, sharex=True)
        fig.suptitle('#{}'.format(self.shot))
        # 微软雅黑,如果需要宋体,可以用simsun.ttc
        myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
        fronts = {'family': 'Times New Roman', 'size': 12}
        for i in range(0, len(self.tag_list), 2):
            time, data = h5r.read_channel(self.shot, channel=self.tag_list[i], device="2a")
            time_t, data_t = h5r.read_channel(self.shot, channel=self.tag_list[i + 1], device="2a")
            axes1 = axes[int(i / 2)]

            lns1 = axes1.plot(time, data, 'r', label=self.tag_list[i])
            axes1.set_ylabel(self.tag_list[i], fontproperties=myfont)
            # axes1.set_yticks(fontproperties='Times New Roman', fontsize=12)
            axes2 = axes1.twinx()
            lns2 = axes2.plot(time_t, data_t, 'b', label=self.tag_list[i + 1])
            axes2.set_ylabel(self.tag_list[i + 1], fontproperties=myfont)
            # 合并图例
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            axes2.legend(lns, labs)
        axes1.set_xlabel('time(s)', fontproperties=myfont)

    def signal_computer(self, time_range):
        if 'EFIT_BETA_N' in self.tag_list:
            time_beta_t, data_beta_t = read_data(self.shot, "EFIT_BETA_T", time_range[0], time_range[1])
            time_bt, data_bt = read_data(self.shot, "BT", time_range[0], time_range[1])
            time_r, data_r = read_data(self.shot, "EFIT_MINOR_R", time_range[0], time_range[1])
            time_ip, data_ip = read_data(self.shot, "IP", time_range[0], time_range[1])
            time, data = read_data(self.shot, "IP", time_range[0], time_range[1] + 0.1)
            beta_N = data_beta_t / (data_ip / (1000 * data_r * data_bt * 0.0622))
            # time, data = h5r.read_channel(self.shot, "IP", device="2a")
            plt.figure()
            myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
            plt.plot(time, data, 'r', label='Ip')
            plt.ylabel('Ip', fontproperties=myfont)
            # axes1.set_yticks(fontproperties='Times New Roman', fontsize=12)
            axes2 = plt.twinx()
            axes2.plot(time_beta_t, beta_N, 'b', label='beta_N')
            axes2.set_ylabel('beta_N', fontproperties=myfont)
            plt.savefig('betaN/{}.png'.format(self.shot), bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    shot_for_shap = [36755, 36823, 37021]
    validshot = np.load(r'LHdataset\H_beta.npy')
    # shot_for_shap = [36755]
    feature_tags = ['IP', "DENSITY", "W_E", "V_LOOP", "FIR01", "FIR03", "DH", "DV", "BOLU03", "BOLU06"]
    sxr_array_t = ['SX01', 'SX02', 'SX03', 'SX04', 'SX05', 'SX06', 'SX07', 'SX08', 'SX09', 'SX10', 'SX11', 'SX12',
                   'SX13', 'SX14', 'SX15', 'SX16', 'SX17', 'SX18', 'SX19', 'SX20']
    bold_u_array_t = ["BOLU03", "BOLU04", "BOLU06", "BOLU07", "BOLU08", "BOLU09", "BOLU10",
                   "BOLU11", "BOLU12", "BOLU13", "BOLU14", "BOLU15", "BOLU16"]
    bold_d_array_t = ["BOLD01", "BOLD02", "BOLD03", "BOLD04", "BOLD05", "BOLD06", "BOLD07", "BOLD08", "BOLD09", "BOLD10",
                    "BOLD11", "BOLD12", "BOLD13", "BOLD14", "BOLD15", "BOLD16"]
    plot2d_bd = Plot2d(bold_d_array_t, 36823)
    plot2d_bd.array_plot()
    plot2d_bu = Plot2d(bold_u_array_t, 36823)
    plot2d_bu.array_plot()
    plot2d_sxr = Plot2d(sxr_array_t, 36823)
    plot2d_sxr.array_plot()

    # plotfea = Plotsignal(feature_tags, 36823)
    # plotfea.signal_plot()
    # for i in range(validshot.shape[0]):
    #     start = h5r.get_attrs("StartTime", shot_number=validshot[i, 0], channel="EFIT_LI")
    #     end = validshot[i, 2] / 1000
    #     plotbetan = Plotsignal(['IP','EFIT_BETA_N'],validshot[i, 0])
    #     plotbetan.signal_computer([start,end])
