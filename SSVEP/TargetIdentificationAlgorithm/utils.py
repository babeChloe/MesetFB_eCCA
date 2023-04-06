import numpy as np
import math

import random
import matplotlib
from sklearn.cross_decomposition import CCA

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 本函数是用于显示类成员函数执行时间。
def timer(func):
    def funcWrapper(*args, **kwargs):
        from time import time
        timeStart = time()
        result = func(*args, **kwargs)
        timeEnd = time()
        timeSpend = timeEnd - timeStart
        print('{} 花费时间 {:5f} s'.format(func.__name__, timeSpend))
        return result

    return funcWrapper


# 绘制脑电图的函数
# EEG=(channel,Timepoint)
def printEEG(EEG, t_task):
    # EEG=(channel,Timepoint)
    nChannel, nTimes = EEG.shape
    teminal = t_task / (nTimes - 1)
    t_list = [i * teminal for i in range(0, nTimes, 1)]
    for channel in range(nChannel):
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        plt.plot(t_list, EEG[channel, :], 'ro-', color=color, alpha=0.8, linewidth=1, label='itr')
    return plt


def sin_wave(freq, nPoints, phase, sfreq=1000):
    """
    构造正弦波函数，用于标准CCA中的正弦模板信号\n
    :param freq: 正弦波频率(float),单位赫兹
    :param nPoints: 采样点的总个数(int)
    :param phase: 正弦波相位因子(float)，范围在0-2，该值会乘以pi
    :param sfreq: 采样频率(float)，单位赫兹，默认是1kHz
    :return: wave (ndarray):(n_points,)正弦波信号
    """
    time_points = np.linspace(0, (nPoints - 1) / sfreq, nPoints)
    # wave = sin(2*pi*freq*time_points + pi*phase)
    wave = np.sin(2 * np.pi * freq * time_points + phase * np.pi)
    return wave


def sine_template(freq, phase, nPoints, nHarmonics, sfreq):
    """
    构建正余弦模板

    Args:
        freq (float or int): Basic frequency.
        phase (float or int): Initial phase.
        nPoints (int): Sampling points.
        nHarmonics (int): Number of harmonics.
        sfreq (float or int): Sampling frequency.

    Returns:
        Y (ndarray): (n_points, 2*n_harmonics).
    """
    Y = np.zeros((nPoints, 2 * nHarmonics))  # (Np, 2Nh)
    for nh in range(nHarmonics):
        Y[:, 2 * nh] = sin_wave((nh + 1) * freq, nPoints, 0 + phase, sfreq)
        Y[:, 2 * nh + 1] = sin_wave((nh + 1) * freq, nPoints, 0.5 + phase, sfreq)
    return Y


def acc_compute(rou):
    """
    计算准确率

    -----------------------------------
    :param rou:所有事件加权相关系数的混淆矩阵(ndarray): (nEvents(real), nEvents(models))
    :return:每个事件的正确分类数(list)，总体准确率(float)
    """
    n_events = rou.shape[0]
    correct = []
    for netr in range(n_events):
        temp = 0
        if np.argmax(rou[netr, :]) == netr:
            temp += 1
        correct.append(temp)
    return correct, np.sum(correct) / n_events


def initialization_based(res):
    idx = res.argmax(axis=1)
    out = np.zeros_like(res, dtype=float)
    out[np.arange(res.shape[0]), idx] = 1
    return out


def cal_itr(number, time, acc):
    """
    计算ITR

    -----------------------------------
    :param number:事件数量
    :param time:秒(float)
    :param acc:0-1(float)
    :return:信息传输率
    """
    part_a = math.log(number, 2)
    if acc == 1.0 or acc == 100:  # avoid spectial situation
        part_b, part_c = 0, 0
    else:
        part_b = acc * math.log(acc, 2)
        part_c = (1 - acc) * math.log((1 - acc) / (number - 1), 2)
    result = 60 / time * (part_a + part_b + part_c)
    return result


def cal_CCA(X, Y):
    """
    StandardCCA count

    ----------------------------------------------------
    :param X: (num of sample points * num of channels1 )
    :param Y: (num of sample points * num of channels2 )
    :return:
    """
    cca = CCA(n_components=1, max_iter=1000)
    # 如果想计算第二主成分对应的相关系数cca = CCA(n_components=2)
    # 训练数据
    cca.fit(X, Y)
    X_train_r, Y_train_r = cca.transform(X, Y)
    corr_cca = np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]
    return corr_cca.real


if __name__ == '__main__':
    pass
