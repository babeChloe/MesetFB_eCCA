import numpy as np

from SSVEP.TargetIdentificationAlgorithm.StandardCCA import BasicCCA
from SSVEP.DataPreprocessing.utils import FiltersUtils
from .utils import timer, sine_template, cal_CCA


class FilterBankCCA(BasicCCA):
    """
    - 滤波器组典型相关分析算法

    -------------------

    - 引用：Chen X., Wang Y., Gao S., et al. Filter Bank Canonical Correlation Analysis for Implementing a HighSpeed SSVEP-Based Brain-Computer Interface[J]. Journal of Neural Engineering, 2015, 12(4):046008.
    - 创建时间: 2022/10/19 15:42
    ------------------

    - 更新日志:
        - 2022/11/01 15：35  Li JiaYi    找到准确率低的罪魁祸首啦（切片！）
        - 2023/3/19 22:14   Li JiaYi    重构项目
    """

    def __init__(self, data, wPass2D, wStop2D, numFilter, freqsList, nHarmonics, fs=1000, useBankflag=False):
        """
        滤波器组分析初始化

        ------------------------------------------------------------------
        :param data: 脑电数据，注意输入的格式为[通道x时间点x事件x试验]
        :param wPass2D: 滤波器通带二维数组
        :param wStop2D: 滤波器阻带二维数组
        :param numFilter: 滤波器个数
        :param freqsList: 正弦波频率(float),单位赫兹
        :param nHarmonics: 正余弦模板谐波数量
        :param fs: 采样频率(float)，单位赫兹，默认是1kHz
        :param useBankflag: 是否对正余弦模板信号进行滤波器组分析，默认为否
        """
        # 滤波器工具
        super().__init__(data, fs)
        self.filterUtil = FiltersUtils()
        # 已滤波的数据
        self.filteredData = self.filterUtil.ChebyshevI_BandpassFilters(data=data, wPass2D=wPass2D, wStop2D=wStop2D,
                                                                       numFilter=numFilter, fs=fs)
        # 子带数量
        self.nBanks = len(self.filteredData)
        # 数据通道
        self.nChannels = self.filteredData['bank1'].shape[0]
        # 时间点数量
        self.nTimes = self.filteredData['bank1'].shape[1]
        # 事件个数
        self.nEvents = self.filteredData['bank1'].shape[2]
        # 试验个数
        self.nTrials = self.filteredData['bank1'].shape[3]
        # 正余弦模板
        self.targetTemplateSet = np.zeros((len(freqsList), 2 * nHarmonics, self.nTimes))
        for nf, freq in enumerate(freqsList):
            self.targetTemplateSet[nf, ...] = sine_template(freq=freq, phase=0, nPoints=self.nTimes,
                                                            nHarmonics=nHarmonics, sfreq=fs).T
        if useBankflag:
            # 根据每个子带都生成正余弦滤波模板
            self.targetTemplateSet = self.filterUtil.ChebyshevI_BandpassFilters(data=self.targetTemplateSet,
                                                                                wPass2D=wPass2D, wStop2D=wStop2D,
                                                                                numFilter=numFilter, fs=fs)
    def fit(self, x_train, y_train):
        pass

    def leave_one(self, selected_trial):
        pass

    @timer
    def predict(self, trial):
        """
        预测

        ------------------------------------------------------------------
        :param trial:试验数据
        :return:所有事件的加权相关系数的混淆矩阵
        """
        bank_id = -1
        # 子带数量 真实事件标注  测试事件数
        # rho = np.zeros((self.nBanks, self.nEvents, self.nEvents))
        res = np.zeros((self.nEvents, self.nEvents))
        if type(self.targetTemplateSet) == dict:
            for bank in self.filteredData:
                data = self.filteredData[bank]
                template = self.targetTemplateSet[bank]
                bank_id += 1  # 记录子带当前id
                test_data = np.transpose(data, [2, 3, 1, 0])
                for event in range(self.nEvents):
                    for eventn in range(self.nEvents):
                        template1 = np.transpose(template[eventn, ...], [1, 0])
                        rr = cal_CCA(test_data[event, trial, ...], template1)
                        w = (bank_id + 1) ** (-1.25) + 0.25
                        res[event, eventn] += w * (rr ** 2)
        else:
            template = self.targetTemplateSet
            for bank in self.filteredData:
                data = self.filteredData[bank]
                bank_id += 1  # 记录子带当前id
                test_data = np.transpose(data, [2, 3, 1, 0])
                for event in range(self.nEvents):
                    for eventn in range(self.nEvents):
                        template1 = np.transpose(template[eventn, ...], [1, 0])
                        rr = cal_CCA(test_data[event, trial, ...], template1)
                        w = (bank_id + 1) ** (-1.25) + 0.25
                        # rho[bank_id, event, eventn]=rr
                        res[event, eventn] += w * (rr ** 2)
        return res


if __name__ == '__main__':
    pass
