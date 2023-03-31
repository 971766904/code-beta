import pandas as pd
import warnings
from scipy import signal
from scipy.fftpack import fft, ifft
from numpy import array, sign, zeros
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy import interpolate
# from DPT.FileOperator import Shot, Signal
from scipy import fftpack
import matplotlib as mpl
import scipy.interpolate as spi
import math
import hdf5Reader2A as h5r
import file_read as fr
from sklearn.datasets import load_breast_cancer,load_boston,load_wine

warnings.filterwarnings("ignore")


# 线性插值函数
def interp_resampling(data, time, freq):
    start_time = 0.01
    down_time = 0.7
    time_new = np.linspace(start_time, down_time, int((down_time - start_time) * freq))
    f = interpolate.interp1d(time, data, kind='linear')
    data_new = f(np.array(time_new))
    return np.array(data_new), np.array(time_new)


def Fun(x, a, b):  # 定义拟合函数形式
    return a * x + b


def intergral_Mirnov(time_raw, data_raw, downtime=0.7):
    f = len(time_raw) / (time_raw[-1] - time_raw[0]) if len(time_raw) > 1 else 0
    dt = 1 / f
    time_fit = time_raw[(time_raw > 0) & (time_raw < downtime)]
    data_fit = data_raw[(time_raw > 0) & (time_raw < downtime)]
    data_int = -np.cumsum(data_fit * dt / 0.069e-4)
    para, pcov = curve_fit(Fun, time_fit, data_int)
    y_fitted = data_int - para[0] * time_fit - para[1]
    return time_fit, y_fitted


def self_filter(data, sampling_freq, low_pass, high_pass):
    """
    -------
    滤波函数
    data为时间序列参数
    sampling_freq为采样频率
    low_pass, high_pass分别为高通与低通频率
    -------
    fixed by C.S.Shen, 2020.12.18
    """
    ba1 = signal.butter(8, (2 * low_pass / sampling_freq), "lowpass")
    filter_data1 = signal.filtfilt(ba1[0], ba1[1], data)
    ba2 = signal.butter(8, (2 * high_pass / sampling_freq), "highpass")
    filter_data = signal.filtfilt(ba2[0], ba2[1], filter_data1)
    return filter_data


def cal_mode_number(data1, data2, time, chip_time=5e-3, down_number=8, low_fre=2e3, high_fre=1e5, step_fre=3e3,
                    max_number=3, var_th=1e-13, real_angle=15, coherence_th=0.95):
    """
    -------
    用于计算模数
    data1，data2为两道Mirnov探针信号（尽可能近）
    time为他们的时间轴

    chip_time为做互功率谱时的窗口时间长度，默认为5ms
    down_number为做互功率谱时的降点数（down_number = 1，即取了全时间窗的点，即FFT窗仅为一个），默认为8
    low_fre为所需最低频率，默认为2kHz
    high_fre为所需最高频率，默认为100kHz
    step_fre为选取最大频率时的步长，默认为3kHz
    max_number为选取最大频率的个数，默认为3个
    var_th为频率间的方差阈值，默认为1e-13
    real_angle为两道Mirnov探针间的极向空间角度差，默认为15°
    coherence_th为互相关系数阈值，默认为0.95
    -------
    C.S.Shen, 2020.12.18
    """
    # set parameters
    dt = time[1] - time[0]  # time interval
    fs = 200 / (time[199] - time[0])  # Sampling frequency
    len_window = int(chip_time * fs)  # window length
    length = len(time)  # length of the window time
    Chip = length // len_window  # number of the chip in the window
    f_low = int(low_fre * chip_time)  # lowest frequency
    f_high = int(high_fre * chip_time)  # Highest frequency
    f_step = int(step_fre * chip_time)  # select max_frequency length
    number_max = int(max_number)  # maximum fre number
    m = []  # m value
    t = []  # time
    fre = []  # frequency
    for t1 in range(Chip):
        t.append(time[len_window * (t1 + 1) - 1])
    # filter
    data1 = self_filter(data1, fs, high_fre, low_fre)
    data2 = self_filter(data2, fs, high_fre, low_fre)
    tmp_var = []
    # slide
    for k in range(Chip):
        if k < Chip - 1:
            chip_data1 = data1[int(len_window * k):int(len_window * k + len_window)] - np.mean(
                data1[int(len_window * k):int(len_window * k + len_window)])
            chip_data2 = data2[int(len_window * k):int(len_window * k + len_window)] - np.mean(
                data2[int(len_window * k):int(len_window * k + len_window)])
        else:
            chip_data1 = data1[int(len_window * k):] - np.mean(data1[int(len_window * k):])
            chip_data2 = data2[int(len_window * k):] - np.mean(data2[int(len_window * k):])

        # calculate cross spectral density
        (f, csd) = signal.csd(chip_data1, chip_data2, fs=fs, window='hann', nperseg=len_window // down_number,
                              scaling='density')
        (f_coherence, coherence) = signal.coherence(chip_data1, chip_data2, fs=fs, window='hann',
                                                    nperseg=len_window // down_number)
        abs_csd = np.abs(csd)
        phase_csd = np.angle(csd) * 180 / np.pi
        angle_csd = np.where(coherence > coherence_th, phase_csd, 0)
        csd_chosen = csd[f_low // down_number:f_high // down_number]
        f_chosen = f[f_low // down_number:f_high // down_number]
        var_csd = np.var(np.abs(csd_chosen))
        tmp_var.append(var_csd)
        ch = np.abs(
            csd_chosen / np.max(np.abs(csd_chosen)))  # 求出归一化之后的互功率谱均值，若最大频率大于最小频率，且小于ch_th就证明存在明显的模式, 目前使用方差与互相关系数做约束
        index_ch = ch.argsort()[::-1][0:number_max]
        # 判断是否有明显模式
        TM_fre = np.zeros(max_number)
        if var_csd < var_th:
            m_t = 0
            TM_fre[0] = 0
        else:
            Piece = f_step // down_number
            index_piece = []
            max_piece = []
            for i in range(len(csd_chosen) // Piece):
                csd_piece = np.abs(csd_chosen[i * Piece:(i + 1) * Piece])
                max_piece.append(max(20 * np.log(csd_piece)))
                index_piece.append(csd_piece.argsort()[::-1][0] + f_low // down_number + i * Piece)
            tmp = np.array(max_piece)
            index_max = tmp.argsort()[::-1][0:number_max]
            N_index_max = len(index_max)
            index = np.zeros(N_index_max, dtype=np.int)
            for ii in range(N_index_max):
                index[ii] = index_piece[index_max[ii]]
            TM_fre = []
            TM_amp = []
            TM_phi = []
            for ind in index:
                TM_fre.append(f[ind])
                TM_amp.append(abs_csd[ind])
                TM_phi.append(angle_csd[ind])
            m_tmp = np.zeros(len(TM_phi), dtype=np.float)
            m_t = 0
            for iii in range(len(TM_phi)):
                if TM_phi[iii] < 0:
                    TM_phi[iii] = TM_phi[iii] + 360
                else:
                    if TM_phi[iii] > 360:
                        TM_phi[iii] = TM_phi[iii] - 360
                m_tmp[iii] = TM_phi[iii] / real_angle * TM_amp[iii] / np.sum(TM_amp)
                m_t = m_t + m_tmp[iii]
        m.append(m_t)
        fre.append(TM_fre[0])
    return t, m, fre


def deg2deg(deg0, deg):
    """
    -------
    change degree from 0 - 360 to -180 - 180
    -------
    C.S.Shen, 2020.10.17
    """
    deg1 = deg0 * 0
    mm = np.size(deg0)
    for mm1 in range(1, mm):
        deg1[mm1] = deg0[mm1] - math.floor((deg0[mm1] + (360 - deg)) / 360) * 360
    return deg1


def n_1_mode(theta, br, deg):
    """
    -------
    用于计算n=1模式幅值与相位（认为不存在高n分量）
    输入为相对角度180°两个锁模探针的位置与数据，deg为其相对角度180°
    输出为求得的n=1幅值与相位
    -------
    br = amp*cos(theta+phase)
    C.S.Shen, 2020.10.17
    """
    theta1 = theta[0] / 180 * math.pi
    theta2 = theta[1] / 180 * math.pi
    D = math.sin(theta1 - theta2)
    br1 = br[0]
    br2 = br[1]
    amp = (br1 ** 2 + br2 ** 2 - 2 * br1 * br2 * math.cos(theta1 - theta2)) ** 0.5 / abs(math.sin(theta1 - theta2))
    cos_phi = (-br2 * math.cos(theta1) + br1 * math.cos(theta2)) / D
    sin_phi = (br2 * math.sin(theta1) - br1 * math.sin(theta2)) / D
    tanPhi = sin_phi / cos_phi
    # phase of origin is -(phs + 2 * pi * f * t)
    # phase of b ^ max is pi / 2 - (phs + 2 * pi * f * t)
    # the variable in sine function
    dlt0 = np.zeros(len(tanPhi), dtype=np.float)
    for i in range(len(tanPhi)):
        dlt0[i] = math.atan(tanPhi[i]) / math.pi * 180 + 180 * np.floor((1 - np.sign(cos_phi[i])) / 2) - 90
    # the variable in cosine function, so it is also the phase of b_theta maximum.
    phase = deg2deg(-dlt0, deg)
    # the phase of b ^ max
    return amp, phase


def locked_mode(shot, time, vbr0, theta=None):
    """
    -------
    用于计算时间序列的n=1模式幅值与相位
    shot为该炮炮号，影响NS值
    time为时间轴
    vbr0为4 * len(time) 的数组，为4个锁模探针的时间序列
    theta为2组对减后的环向空间角度
    -------
    C.S.Shen, 2020.12.18
    """
    if theta is None:
        theta = [67.5, 157.5]
    tau_br = [10e-3, 10e-3, 10e-3, 10e-3]
    br_Saddle = np.zeros((4, len(time)), dtype=np.float)
    for j1 in range(len(tau_br)):
        br_Saddle[j1] = vbr0[j1] / tau_br[j1] * 1e4
    br_odd = np.zeros((2, len(time)), dtype=np.float)  # 创建2维数组存放诊断数据
    amp = np.zeros(len(time), dtype=np.float)
    phase = np.zeros(len(time), dtype=np.float)
    br_odd[0] = br_Saddle[0] - br_Saddle[2]
    br_odd[1] = br_Saddle[1] - br_Saddle[3]
    amp, phase = n_1_mode(theta, br_odd, 180)
    return amp, phase


def factor_time_series(data_slice):
    """
    -------
    用于计算时间序列偏度、峰度、与方差
    也可用于锯齿特征提取
    -------
    C.S.Shen, 2020.12.18
    """
    skew = []
    kurt = []
    var = []
    N = len(data_slice)
    for i in range(N):
        skew.append(pd.Series(data_slice[i]).skew())
        kurt.append(pd.Series(data_slice[i]).kurt())
        var.append(np.var(data_slice[i]))
    return skew, kurt, var


# 中位点采样的信号进行加工处理
def mean_down_sampling(data, time, freq):
    fs = len(data) / (time[-1] - time[0]) if len(time) > 1 else 0
    fs_int = round(fs / 1000) * 1000
    len_chip = fs_int / freq
    processed_data = []
    processed_time = []
    section = int((len(data) - len_chip) / len_chip) + 1
    for j in range(section):
        if j < (section - 1):
            mean_data = np.mean(data[int(len_chip * j):int(len_chip * j + len_chip)])
            middle_time = time[int(len_chip * j + len_chip / 2)]
            processed_data.append(mean_data)
            processed_time.append(middle_time)
        else:
            Templen = len(data[int(len_chip * j):])
            mean_data = np.mean(data[int(len_chip * j):])
            middle_time = time[int(len_chip * j + Templen / 2)]
            processed_data.append(mean_data)
            processed_time.append(middle_time)
    return np.array(processed_data), np.array(processed_time)


# 中位点采样的信号进行加工处理
def middle_down_sampling(data, time, freq):
    fs = len(data) / (time[-1] - time[0]) if len(time) > 1 else 0
    chiplen = fs / freq
    processed_data = []
    processed_time = []
    section = math.ceil(len(data) / chiplen)
    for j in range(section):
        if j < (section - 1):
            middle_data = data[int(chiplen * j + chiplen / 2)]
            middle_time = time[int(chiplen * j + chiplen / 2)]
            processed_data.append(middle_data)
            processed_time.append(middle_time)
        else:
            Templen = len(data[int(chiplen * j):])
            middle_data = data[int(chiplen * j + Templen / 2)]
            middle_time = time[int(chiplen * j + Templen / 2)]
            processed_data.append(middle_data)
            processed_time.append(middle_time)
    return np.array(processed_data), np.array(processed_time)


def factors_profile(data, time, n_channel, fs):
    """
    -------
    用于计算存在剖面信号的空间偏度、峰度与方差
    data为n_channel * len(time)维度的时间序列
    n_channel为该诊断的通道数
    fs为预期采样率，要低于真实采样率
    如果fs为-1则证明无需降采样
    -------
    C.S.Shen, 2020.12.18
    """
    fr = data.shape[1] / (time[-1] - time[0]) if len(time) > 1 else 0
    fr_int = round(fr / 1000) * 1000
    if fs < 0:
        skew = np.zeros(len(time), dtype=np.float)
        kurt = np.zeros(len(time), dtype=np.float)
        var = np.zeros(len(time), dtype=np.float)
        data = np.array(data)
        for j in range(len(time)):
            skew[j] = pd.Series(data[:, j]).skew()
            kurt[j] = pd.Series(data[:, j]).kurt()
            var[j] = np.var(data[:, j])
        time_down = time
    else:
        time_down = []
        data_down = np.zeros((int(n_channel), int(fs * data.shape[1] / fr_int)), dtype=np.float)
        for i in range(n_channel):
            data_down[i], time_down = mean_down_sampling(data[i], time, fs)
        skew = np.zeros(len(time_down), dtype=np.float)
        kurt = np.zeros(len(time_down), dtype=np.float)
        var = np.zeros(len(time_down), dtype=np.float)
        for j in range(len(time_down)):
            skew[j] = pd.Series(data_down[:, j]).skew()
            kurt[j] = pd.Series(data_down[:, j]).kurt()
            var[j] = np.var(data_down[:, j])
    return skew, kurt, var, time_down


def time_slice(time, data, noverlap=0.5, slice_length=10):
    data_length = len(data)
    data_slice = []
    time_slice = []
    nstep = noverlap * slice_length
    Nwindows = math.floor((data_length / slice_length - noverlap) / (1 - noverlap))
    for i in range(Nwindows):
        unit = data[int(data_length - 1 - (Nwindows - i) * nstep - (slice_length - 1)):int(
            data_length - 1 - (Nwindows - i) * nstep)]
        unit_time = time[int(data_length - 1 - (Nwindows - i) * nstep)]
        data_slice.append(unit)
        time_slice.append(unit_time)
    return time_slice, data_slice


def it2bt(data):
    """
    -------
    EAST装置用于计算Bt
    data为it
    -------
    C.S.Shen, 2021.11.03
    """
    Bt = (4 * math.pi * 1e-7) * data * (16 * 130) / (2 * math.pi * 1.8)
    return Bt


def fft_axk(data, time, chip_length=5e-3):
    fs = len(data) / (time[-1] - time[0])
    N = int(chip_length * fs)
    time_window_len = int(chip_length * fs)
    FT_time = []
    FT_abs = []
    FT_freq = []
    for i in range(len(data)):
        if i >= time_window_len:
            FT_time.append(time[i])
            time_window = data[(i - time_window_len): i]
            fft_data = fft(time_window)
            abs_datafft = np.abs(fft_data)
            abs1 = abs_datafft[0] / N
            abs2 = abs_datafft[1:] * 2 / N
            abs_datafft = np.hstack((abs1, abs2))
            angle_datafft = np.angle(fft_data)
            freq = [n * fs / (N - 1) for n in range(N)]
            max_index = np.argmax(abs_datafft)
            corr_freq = freq[int(max_index)]
            corr_abs = abs_datafft[max_index]
            FT_abs.append(corr_abs)
            FT_freq.append(corr_freq)
    return FT_time, FT_abs, FT_freq


def envelope(data):
    index = list(range(len(data)))
    # 获取极值点
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    # 将极值点拟合为曲线
    ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)  # 样本点导入，生成参数
    iy3_max = spi.splev(index, ipo3_max)  # 根据观测点和样条参数，生成插值

    ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)  # 样本点导入，生成参数
    iy3_min = spi.splev(index, ipo3_min)  # 根据观测点和样条参数，生成插值
    '''
    f_max = interpolate.interp1d(max_peaks, data[max_peaks], kind='linear')
    iy3_max = f_max(time)
    f_min = interpolate.interp1d(min_peaks, data[min_peaks], kind='linear')
    iy3_min = f_min(time)
    '''
    # 计算平均包络线
    iy3_mean = (iy3_max + iy3_min) / 2
    return iy3_max, iy3_min, iy3_mean


def subtract_drift(data, time):
    index = np.where(time < 0)
    mean = np.mean(data[index])
    data = data - mean
    return data


def smooth2nd(data, M):  # x 为一维数组
    K = round(M / 2 - 0.1)  # M应为奇数，如果是偶数，则取大1的奇数
    lenX = len(data)
    data_smooth = np.zeros(lenX)
    if lenX < 2 * K + 1:
        print('数据长度小于平滑点数')
        data_smooth = data
    else:
        for NN in range(0, lenX, 1):
            startInd = max([0, NN - K])
            endInd = min(NN + K + 1, lenX)
            data_smooth[NN] = np.mean(data[startInd:endInd])
    return data_smooth


def cal_p_rad(data_1, data_2, time, n_channel, fs):
    if fs < 0:
        data_down_1 = np.array(data_1)
        data_down_2 = np.array(data_2)
        time_down = time
    else:
        time_down = []
        data_down_1 = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
        data_down_2 = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
        for i in range(n_channel):
            data_down_1[i], time_down = middle_down_sampling(data_1[i], time, fs)
            data_down_2[i], time_down = middle_down_sampling(data_2[i], time, fs)
    KVT = np.array(
        [0.72, 0.7597, 0.7730, 0.7611, 0.7921, 0.8373, 0.8911, 0.9666, 1.0000, 0.9409, 0.9851, 0.9805, 0.9443, 0.9264,
         0.9084, 0.7])
    KVD = np.array(
        [0.66, 0.6823, 0.6870, 0.6703, 0.6716, 0.7158, 0.7593, 0.9000, 0.8686, 0.7979, 0.8607, 0.8864, 0.9073, 0.7865,
         0.8732, 0.85])
    rvt = np.array([-18, 2.5, 22, 42, 64, 85, 105, 127, 147.5, 168, 189, 208.5, 227.5, 246.5, 264.5, 282]) * -1
    rvd = np.array([-19, 1.5, 23, 43, 66, 89, 110.5, 130, 151, 171.5, 193, 212, 232, 251, 268, 286])
    smo = 5  # smo表征对信号的平均点数，AXUV采样率为50K，smo越大，则一些高频现象会看不到；但是太小，信号又不是很好，所以根据需求定义
    R = 1.05  # 大半径单位m
    Aap = 0.0008 * 0.005  # 狭缝面积，单位m ^ 2
    Adet = 0.002 * 0.005  # 探测器面积，单位m ^ 2
    dxf = 0.054  # 探测器到狭缝距离，单位m
    PAi = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
    PFi = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
    ya = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
    yb = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
    for j in range(n_channel):
        PAi[j] = smooth2nd(data_down_1[j] / 0.26, smo)
        PFi[j] = smooth2nd(data_down_2[j] / 0.26, smo)
        if j <= n_channel // 2:
            ya[j] = PAi[j] / KVT[j] / 2e5  # 200000是放大器倍数，前八道是2 * 10 ^ 5
            yb[j] = PFi[j] / KVD[j] / 2e5  # 200000是放大器倍数，前八道是2 * 10 ^ 5
        else:
            ya[j] = PAi[j] / KVT[j] / 5e5  # 500000是放大器倍数，后八道是5 * 10 ^ 5
            yb[j] = PFi[j] / KVD[j] / 5e5  # 500000是放大器倍数，后八道是5 * 10 ^ 5
    ya[0] = ya[1] * 0.98
    yb[1] = 0.5 * (yb[0] + yb[2])
    index = np.where(time_down < 0)
    for j in index:
        k = ya[1][j] / yb[1][j]
        for i in range(n_channel):
            ya[i][j] = yb[i][j] * k  # 两阵列校准
    sumv1, sumv2, sumv3, sumv4, sumv5 = 0, 0, 0, 0, 0
    for j in range(n_channel - 1):
        i = j + 1
        sumv1 = (rvt[i] * ya[i]) + (rvd[i] * yb[i]) + sumv1
        if i == 1:
            sumv2 = sumv2  # 高场侧
            sumv3 = np.abs(rvt[i] - rvd[i]) * ya[i] + sumv3  # 低场侧
            sumv5 = np.abs(rvt[i] - rvd[i]) * ya[i] * rvt[i] + sumv5  # 一个位置权重
        else:
            sumv2 = np.abs(rvt[i] - rvt[i - 1]) * ya[i] + sumv2
            sumv3 = np.abs(rvd[i] - rvd[i - 1]) * yb[i] + sumv3
            sumv5 = np.abs(rvt[i] - rvt[i - 1]) * ya[i] * rvt[i] + abs(rvd[i] - rvd[i - 1]) * yb[i] * rvd[i] + sumv5
        sumv4 = sumv4 + ya[i] + yb[i]
    sumvt = (sumv2 + sumv3) / 1e6
    P_rad = sumvt * (4 * np.pi * dxf ** 2) / (Adet * Aap) * (2 * np.pi * R) / 2.2
    return P_rad, time_down


def sum_ne(time, data, n_channel, fs):
    if fs < 0:
        data_down = np.array(data)
        time_down = time
    else:
        time_down = []
        data_down = np.zeros((int(n_channel), int(np.round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
        for i in range(n_channel):
            data_down[i], time_down = middle_down_sampling(data[i], time, fs)
    A = np.zeros(len(time_down), dtype=np.float)
    for i in range(len(time_down)):
        A[i] = np.sum(data_down[:, i])
    return A, time_down


def cal_p_total(ip_raw, vl_raw, time_ip, time_vl):
    ip, time = interp_resampling(ip_raw, time_ip, 10e3)
    vl, time = interp_resampling(vl_raw, time_vl, 10e3)
    P_tot = np.multiply(ip, vl)
    return P_tot, time


def time_win_old(data, time, win_size, step):
    fs = int(len(data) / (time[-1] - time[0]))
    sample_rate = round(fs / 1000) * 1000
    index = 0
    X, time_win = [], []
    while int((index + win_size) * sample_rate) <= data.shape[0]:
        sample = data[int(index * sample_rate):int((index + win_size) * sample_rate)]
        X.append(sample)
        time_win.append(time[int((index + win_size) * sample_rate) - 1])
        index += step
    X.append(data[int(-win_size * sample_rate):])
    time_win.append(time[-1])
    return X, time_win, sample_rate


def time_win(data, time, win_size, step):
    fs = int(len(data) / (time[-1] - time[0]))
    sample_rate = round(fs / 1000) * 1000
    win_size = int(win_size * sample_rate)
    step = int(step * sample_rate)
    index = 0
    X, time_win = [], []
    while (index + win_size) <= data.shape[0]:
        sample = data[index:(index + win_size)]
        X.append(sample)
        time_win.append(time[index + win_size - 1])
        index += step
    X.append(data[-win_size:])
    time_win.append(time[-1])
    return X, time_win, sample_rate


def time_win_start(data, time, win_size, step):
    fs = int(len(data) / (time[-1] - time[0]))
    sample_rate = round(fs / 1000) * 1000
    num_data = int((len(data) - win_size * sample_rate) / (step * sample_rate)) + 1
    start_index = int(len(data) - ((num_data - 1) * step * sample_rate + win_size * sample_rate))
    index = 0
    X, time_win = [], []
    while int((index + win_size) * sample_rate) <= data.shape[0]:
        sample = data[int(start_index + index * sample_rate):int(start_index + (index + win_size) * sample_rate)]
        X.append(sample)
        time_win.append(time[int(start_index + (index + win_size) * sample_rate) - 1])
        index += step
    return X, time_win, sample_rate


def mean_win_data(win_data):
    point_data = []
    for i in range(len(win_data)):
        point_data.append(win_data[i].mean())
    return point_data


def freq_data(win_data, fs):
    import freqdomain
    freq_main_data = []
    for i in range(len(win_data)):
        freqpro = freqdomain.freqdomain(win_data[i].reshape(1, len(win_data[i])), fs)
        freq_main = freqpro.main_freq(percent1=0.3, percent2=0.5, percent3=0.9)
        freq_main_data.append(freq_main)
    return freq_main_data


def fft_zy(win_data, fs):
    N = len(win_data[0])
    FT_abs = []
    FT_freq = []
    for i in range(len(win_data)):
        fft_data = fft(win_data[i])
        abs_datafft = np.abs(fft_data)
        abs1 = abs_datafft[0] / N
        abs2 = abs_datafft[1:] * 2 / N
        abs_datafft = np.hstack((abs1, abs2))
        angle_datafft = np.angle(fft_data)
        freq = [n * fs / (N - 1) for n in range(N)]
        max_index = np.argmax(abs_datafft)
        corr_freq = freq[int(max_index)]
        corr_abs = abs_datafft[max_index]
        FT_abs.append(corr_abs)
        FT_freq.append(corr_freq)
    return np.array(FT_abs), np.array(FT_freq)


def spec_fig(data, fs):
    f, t, Sxx = signal.spectrogram(data, fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


if __name__ == '__main__':
    time, data = fr.read_data(36171, 'SX06', 0.0309999, 1.886)
    timeip, dataip = fr.read_data(36171, "IP", 0.0309999, 1.886)
    time_m, data_mp13 = fr.read_data(36171, 'MP13', 0.0309999, 1.886)
    X_mir, index1, fs1 = time_win(data_mp13, time_m, win_size=0.030, step=0.001)
    X_ip, index2, fs2 = time_win(dataip, timeip, win_size=0.030, step=0.001)
    # X_st_sx, indexsx, fs2 = time_win_start(dataip, timeip, win_size=0.030, step=0.001)
    X_st_ip, indexip, fs2 = time_win(data, time, win_size=0.030, step=0.001)
    # X_ip = mean_win_data(X_ip)
    # X_mir = mean_win_data(X_mir)
    # X_fre = freq_data(X_mir, 100000)
    # X_fft = fft_zy(X_mir, 100000)
    # spec_fig(X_mir[232],100000)

