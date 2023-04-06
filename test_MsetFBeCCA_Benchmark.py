import numpy as np

from SSVEP.TargetIdentificationAlgorithm.utils import acc_compute, cal_itr
from SSVEP.DataPreprocessing import FiltersUtils, cutting_data, Benchmark
from SSVEP.TargetIdentificationAlgorithm import MsetFB_eCCA

if __name__ == '__main__':
    # 文件路径
    dataset_path = 'Benchmark'
    # 取S1.mat测试代码
    data_path = 'Benchmark/S1.mat'
    # 受试者ID:S1
    subject_id = ['S' + '{:01d}'.format(idx_subject + 1) for idx_subject in range(1)]
    # 采样率
    fs = 250
    # 谐波数量
    n_harmonics = 5
    # 视觉延迟时间
    t_delay = 0.14
    # 视觉反映时间
    t_reaction = 0.5
    # 测试时间长度
    t_task = 0.5
    # 通道信息
    chans = ['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    # 滤波器组处理参数
    filter_nums = 5
    w_pass_2d = np.array([[6, 14, 22, 30, 38, 46, 54, 62, 70, 78], [90, 90, 90, 90, 90, 90, 90, 90, 90, 90]])
    w_stop_2d = np.array([[4, 12, 20, 28, 36, 44, 52, 60, 64, 72], [92, 92, 92, 92, 92, 92, 92, 92, 92, 92]])
    # 加载数据
    benchmarkTool = Benchmark()
    freqsList, phaseList = benchmarkTool.get_freqs_and_phases(dataset_path=dataset_path)

    # 取1号受试者数据、
    print("取1号受试者数据做代码测试")
    data = benchmarkTool.load_single_subject_data(filePath=data_path, chans=chans)

    # %% 数据预处理
    # 数据切片
    data = cutting_data(data, t_delay, t_reaction, t_task, fs)
    [nChannels, nTimes, nEvents, nTrials] = data.shape

    # test algorithm: MsetFB_eCCA
    test = MsetFB_eCCA(data=data, wPass2D=w_pass_2d, wStop2D=w_stop_2d, numFilter=filter_nums, freqsList=freqsList,
                       nHarmonics=n_harmonics, fs=fs)

    acc_all = []
    itr_all = []
    tmpacc = 0
    for trial in range(nTrials):
        print('-' * 50)
        print('正在测试第{}个试验数据'.format(trial + 1))
        test.leave_one(trial)
        test.fit()
        res = test.predict(trial)
        _, t = acc_compute(res)
        tmpacc += t
    Acc = tmpacc / nTrials
    acc_all.append(Acc)
    itr = cal_itr(nEvents, t_task, Acc)
    itr_all.append(itr)
    print('-' * 50)
    print('准确率为{:05f}'.format(Acc))
    print('ITR=', itr)
