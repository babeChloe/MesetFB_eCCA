import numpy as np

from SSVEP.TargetIdentificationAlgorithm.StandardCCA import BasicCCA
from .utils import timer, cal_CCA


class MutiSetCCA(BasicCCA):
    """
    - 多子集典型相关分析算法

    -------------------

    - 引用：Zhang Y , Zhou G , Jin J , et al. Frequency Recognition in SSVEP-based BCI using Multiset Canonical Correlation Analysis[J]. 2013.
    - 创建时间: 2022/10/26 19:53
    ------------------

    - 更新日志:
        - 2022/11/01 15：35  Li JiaYi    找到了罪魁祸首，归一化！！！！！
        - 2023/3/19 22:14   Li JiaYi    重构项目
    """
    def __init__(self, data, fs=1000):
        """
        多子集CCA

        -------------------------------------------------
        :param data: 脑电数据，注意输入的格式为[通道x时间点x事件x试验]
        :param fs: 采样频率(float)，单位赫兹，默认是1kHz
        """
        super().__init__(data, fs)
        [self.nChannels, self.nTimes, self.nEvents, self.nTrials] = self.data.shape
        self.trainData = np.zeros((self.nChannels, self.nTimes, self.nEvents, self.nTrials - 1))
        self.testData = np.zeros((self.nChannels, self.nTimes, self.nEvents, 1))
        self.template = list()

    def leave_one(self, selectedTrial):
        """
        留一交叉验证

        -----------------------------------
        :param selectedTrial: 被选作为测试数据的试验
        """
        index = 0
        for i in range(self.nTrials):
            if selectedTrial == i:
                self.testData[:, :, :, 0] = self.data[:, :, :, selectedTrial]
            else:
                self.trainData[:, :, :, index] = self.data[:, :, :, i]
                index += 1

    def zca_whitening(self, inputs):
        """
        ZCA白化

        引用：https://zhuanlan.zhihu.com/p/414275930

        --------------------------------------------
        :param inputs:需要白化的数据
        :return:ZCAMatrix,白化矩阵向量
            np.dot(ZCAMatrix, inputs)，白化后的数据
        """
        sigma = np.dot(inputs, inputs.T) / inputs.shape[
            0]
        # inputs是经过归一化处理的，所以这边就相当于计算协方差矩阵
        U, S, V = np.linalg.svd(sigma)
        # 奇异分解
        epsilon = 0.1
        # 白化的时候，防止除数为0
        A = np.diagonal(1.0 / np.sqrt(np.diag(S) + epsilon))
        A = np.diag(A)
        ZCAMatrix = np.dot(np.dot(U, A), U.T)
        # ZCAMatrix=np.dot(U.T,A)
        # 计算zca白化矩阵
        return ZCAMatrix, np.dot(ZCAMatrix, inputs)

    @timer
    def fit(self):
        """
        训练数据获得模板信号

        --------------------------------------------
        """
        # 输入的数据是(channel,timepoint,40,5)
        # 对应事件下的数据
        self.template.clear()
        TrainData = np.transpose(self.trainData, [2, 3, 0, 1])
        [nEvents, nTrials, nChannels, nTimes] = TrainData.shape
        X = np.zeros((nTrials, nChannels, nTimes))
        for event in range(nEvents):
            # 1. 对脑电数据进行ZCA白化处理
            for trial in range(nTrials):
                Xtemp = TrainData[event, trial, ...]
                # printEEG(Xtemp,2)
                Xtemp = Xtemp - np.tile(np.mean(Xtemp, 0), (Xtemp.shape[0], 1))  # 在列方向上重复对行取平均值9次，行1次
                # printEEG(Xtemp, 2)
                V, Xtemp = self.zca_whitening(Xtemp)
                # printEEG(Xtemp, 2)
                X[trial, :, :] = Xtemp
            # 2. 组合R、S矩阵
            Y = np.zeros((nChannels, nTimes))
            for trial in range(nTrials):
                if trial == 0:
                    Y = X[trial, :, :]
                else:
                    Y = np.vstack((Y, X[trial, :, :]))
            R = np.cov(Y)
            S = np.diag(np.diag(R))
            S_ = np.linalg.inv(S)

            # 3. 求解广义逆
            A = S_ @ (R - S)
            e_val, e_vec = np.linalg.eig(A)
            W = e_vec[:, [np.argmax(e_val)]].T
            W = np.resize(W, (nTrials, nChannels))
            Y = np.zeros((nTrials, nTimes))
            for trial in range(nTrials):
                W1 = W[trial, :]
                X1 = X[trial, ...]
                Y[trial, :] = W1 @ X1
            self.template.append(Y)

    @timer
    def predict(self):
        """
        分类预测

        -----------------------------------
        :return:所有事件的加权相关系数的混淆矩阵
        """
        res = np.zeros((self.nEvents, self.nEvents))
        test_data = np.transpose(self.testData, [2, 3, 1, 0])
        for event in range(self.nEvents):
            for eventn in range(self.nEvents):
                template1 = np.transpose(self.template[eventn], [1, 0])
                # template1 = self.template[eventn]
                rr1 = cal_CCA(test_data[event, 0, ...], template1)
                res[event, eventn] += rr1
        return res


if __name__ == '__main__':
    pass
