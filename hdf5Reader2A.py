# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 9:41
# @Author  : yaoh
# @FileName: my_hdf5.py
# @Software: PyCharm Community Edition
# @email   : hyao666@foxmail.com
# 这代码不能用
import h5py
import numpy as np

from config import hdf5ReaderConfig2A


def read_channel(shot_number, channel="IP", device="2a"):
    shot_number = int(shot_number)
    # 读取某个通道的值及对应时间
    value = if_channel_exist(shot_number, channel, device)[channel]

    # 读取信号时间信息
    t_start = get_attrs("StartTime", value)
    t_freq = get_attrs("SampleRate", value)

    value = np.array(value).reshape(value.shape[0])
    channel_length = value.shape[0]
    if channel[0:4] == "EFIT":
        if t_freq != 1000:
            t_freq = 1000
        time = t_start + np.arange(0, channel_length) / t_freq
        # time = np.round(time,4)
    else:
        time = t_start/1000 + np.arange(0, channel_length) / (t_freq*1000)
    return time, value


def read_file(shot_number, device="2a"):
    """
    读取hdf5文件
    :param shot_number: 炮号
    :param device: 装置名（不区分大小写）
    :return: hdf5 对象
    """
    shot_number = int(shot_number)

    dir_name = shot_number // 200 * 200

    device = device.upper()
    device = hdf5ReaderConfig2A.device_name[device]

    # 这个地方会用到dir_name,由于是用eval，所以不显式表明引用
    file_dir = eval(hdf5ReaderConfig2A.file_dir[device])

    # 加载hdf5文件
    data = h5py.File(file_dir, "r")

    return data


def if_channel_exist(shot_number, channel="IP", device="2a"):
    """

    :param shot_number:
    :param channel:
    :param device:
    :return: HDF5 group，并不读取内部数据，但是可以通过output[channel]获取hdf5对象
    """
    # 仅检查通道是否存在，不读取其中数据
    shot_number = int(shot_number)
    Data = read_file(shot_number, device)

    try:
        # 获取目录
        channel_dir = hdf5ReaderConfig2A.dir[channel]
        output = eval(channel_dir)

        if channel in output.keys():
            return output
        else:
            return False

    except KeyError or AssertionError:
        print(str(shot_number) + " " + channel + " doesn't exist")
        return False


def get_attrs(attributes, value=None, **kwargs):
    """

    :param attributes:  要读取的属性
    :param value:       对应通道
    :param kwargs:      可以直接输入shot_number，channel，device查询
    :return:            通道对应属性
    example:
        Freq = mh.get_attrs("T_Freq", shot_number=20000, channel="HMode_Ha")
        Freq:10000
    """
    # 获取通道的属性，value是读取到的通道
    if value is None:
        shot_number = kwargs['shot_number']
        value = kwargs['channel']
        try:
            device = kwargs['device']
        except KeyError:
            device = "2A"
            value = if_channel_exist(shot_number, value, device)[value]

    return value.attrs[attributes]
