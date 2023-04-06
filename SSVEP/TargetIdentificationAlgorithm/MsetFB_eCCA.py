import numpy as np

from SSVEP.TargetIdentificationAlgorithm.FBCCA import FilterBankCCA
from .MsetCCA import MutiSetCCA
from SSVEP.TargetIdentificationAlgorithm.eCCA import eCCA
from .utils import timer, cal_CCA


class MsetFB_eCCA(FilterBankCCA):
    """
    -MsetCCA+FBCCA+eCCA的集成算法

    """

    def __init__(self, data, wPass2D, wStop2D, numFilter, freqsList, nHarmonics, fs=1000, useBankflag=False):
        """

        :param data: 脑电数据，格式为[通道x时间点x事件x试验]
        :param wPass2D: 滤波器通带二维数组
        :param wStop2D: 滤波器阻带二维数组
        :param numFilter: 滤波器个数
        :param freqsList: 正弦波频率(float),单位赫兹
        :param nHarmonics: 正余弦模板谐波数量
        :param fs: 采样频率(float)，单位赫兹，默认是1kHz
        :param useBankflag: 是否对正余弦模板信号进行滤波器组分析，默认为否
        """
        super().__init__(data, wPass2D, wStop2D, numFilter, freqsList, nHarmonics, fs,
                         useBankflag=False)
        self.trainData = {}
        self.testData = {}
        self.template_M = list()  # MsetCCA模板
        self.template_avg = np.zeros((self.nChannels, self.nTimes, self.nEvents))  # 个体平均模板

    def leave_one(self, selected_trial):
        """
        留一交叉验证

        ----------------------------------------
        :param selected_trial: 被选作为测试数据的试验
        """
        for bank in self.filteredData:
            index = 0
            data = self.filteredData[bank]
            tmp_train_data = np.zeros((self.nChannels, self.nTimes, self.nEvents, self.nTrials - 1))
            tmp_test_data = np.zeros((self.nChannels, self.nTimes, self.nEvents, 1))
            for i in range(self.nTrials):
                if i == selected_trial:
                    tmp_test_data[:, :, :, 0] = data[:, :, :, i]
                    self.testData[bank] = tmp_test_data
                else:
                    tmp_train_data[:, :, :, index] = data[:, :, :, i]
                    self.trainData[bank] = tmp_train_data
                    index += 1

    def avg_temp(self, data):
        """
        获取个体平均模板

        ------------------------
        """
        self.template_avg = data.mean(axis=3)
        self.template_avg = np.transpose(self.template_avg, [2, 1, 0])  # (nEvents, nTimes, nChannels)

    def fit(self):
        """
        训练数据得到模板

        --------------------
        """
        # 获取MsetCCA模板
        self.MutiSetCCA = MutiSetCCA(self.trainData['bank1'], self.fs)
        self.MutiSetCCA.trainData = self.trainData['bank1']
        self.MutiSetCCA.fit()
        self.template_M = self.MutiSetCCA.template
        # 获取个体平均模板
        self.avg_temp(self.trainData['bank1'])

    def _combine_feature(self, X):
        """
        集成分类器，用于结合多个相关系数得到最终特征值

        -----------------------------------
        :param X: List of one-level features.
        :return: Two-level feature.
        """
        tl_feature = 0
        for feature in X:
            sign = abs(feature) / feature
            tl_feature += sign * (feature ** 2)
        return tl_feature

    @timer
    def predict(self, trial):
        """
        分类预测

        ------------------------------
        :param trial:正在被测试的试验
        :return: 所有事件的加权相关系数的混淆矩阵
        """
        bank_id = -1
        res = np.zeros((self.nEvents, self.nEvents))
        transpose = np.transpose
        template1 = transpose(self.targetTemplateSet, [0, 2, 1])
        template2 = transpose(self.template_M, [0, 2, 1])
        template3 = self.template_avg
        for bank in self.filteredData:
            data = self.testData[bank]
            bank_id += 1
            test_data = transpose(data, [2, 3, 1, 0])
            for event_t in range(self.nEvents):
                for event_p in range(self.nEvents):
                    t1 = template1[event_p]
                    t2 = template2[event_p]
                    t3 = template3[event_p]
                    r1 = cal_CCA(test_data[event_t, 0, ...], t1)
                    r2 = cal_CCA(test_data[event_t, 0, ...], t2)
                    r3 = cal_CCA(test_data[event_t, 0, ...], t3)
                    r = self._combine_feature([r1, r2, r3])
                    w = (bank_id + 1) ** (-1.25) + 0.25
                    rr = w * (r ** 2)
                    res[event_t, event_p] += rr
        return res


if __name__ == '__main__':
    pass
