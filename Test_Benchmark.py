import os
import numpy as np
import scipy.io as sio
import xlsxwriter as xw

from SSVEP.TargetIdentificationAlgorithm.utils import acc_compute, cal_itr
from SSVEP.DataPreprocessing import FiltersUtils, cutting_data, Benchmark
from SSVEP.TargetIdentificationAlgorithm.FBCCA import FilterBankCCA
from SSVEP.TargetIdentificationAlgorithm.eCCA import eCCA
from SSVEP.TargetIdentificationAlgorithm.MsetCCA import MutiSetCCA
from SSVEP.TargetIdentificationAlgorithm.MsetFB_eCCA import MsetFB_eCCA

if __name__ == '__main__':
    # %% 参数设置
    # 数据文件路径
    filepath = 'Benchmark'
    # 结果文件路径
    res_path = 'Results'
    # 受试者ID
    subject_id = ['S' + '{:01d}'.format(idx_subject + 1) for idx_subject in range(35)]
    # 采样率
    fs = 250
    # 谐波数量
    n_harmonics = 5
    # 视觉延迟时间
    t_delay = 0.14
    # 视觉反应时间
    t_reaction = 0.5
    # 测试时间长度
    # t_task = 0.5
    # 测试时间片
    t_task_list = [i * 0.25 for i in range(2, 11, 1)]
    # 选取通道
    chans = ['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    # 滤波器组处理参数
    filter_nums = 5
    w_pass_2d = np.array([[6, 14, 22, 30, 38, 46, 54, 62, 70, 78], [90, 90, 90, 90, 90, 90, 90, 90, 90, 90]])
    w_stop_2d = np.array([[4, 12, 20, 28, 36, 44, 52, 60, 64, 72], [92, 92, 92, 92, 92, 92, 92, 92, 92, 92]])

    # %% 加载数据
    benchmarkTool = Benchmark()
    # 获取数据集
    Data = benchmarkTool.load_subjects_data(subjectList=subject_id, filePath=filepath, chans=chans)
    # 获取数据集的频率和相位
    freqsList, phaseList = benchmarkTool.get_freqs_and_phases(dataset_path=filepath)

    # %% 创建存放results的工作簿
    workbook = xw.Workbook(res_path + 'result.xlsx')

    # %% 导入算法
    # method_list = ['eCCA', 'MsetCCA', 'FBCCA', 'MsetFB_eCCA']
    method_list = ['FBCCA', 'eCCA', 'MsetCCA', 'MsetFB_eCCA']

    # %% 测试算法
    for method in method_list:
        print('%' * 80)
        print('当前测试{}'.format(method))
        # 创建子表并激活
        worksheet1 = workbook.add_worksheet(method)
        worksheet1.activate()
        write_point = 1
        # 创建结果存放文件夹
        save_res_path = res_path + '/' + method
        if not os.path.exists(save_res_path):
            os.makedirs(save_res_path)  # 递归创建目录

        # 开始测试
        for t_task in t_task_list:
            print('$' * 80)
            print('当前测试时间片为{}'.format(t_task))
            # 正确率存放数组
            acc_all = []
            # ITR存放数组
            itr_all = []
            # 文件名命名
            if method in ['eCCA', 'MsetCCA']:
                filename = method + '(t_task={})_Result.mat'.format(t_task)
            else:
                filename = method + '(t_task={}, filter_num={})_Result.mat'.format(t_task, filter_nums)

            for id in subject_id:
                print('+' * 80)
                print('当前测试数据集为{}'.format(id))
                org_data = Data[id]

                #  %% 数据切片
                # 未滤波的数据(for FBCCA)
                data = cutting_data(org_data, t_delay, t_reaction, t_task, fs)
                # 高通滤波后的数据
                filtersUtils = FiltersUtils()
                org_filtered_data = filtersUtils.ChebyshevI_BandpassFilters(data, w_pass_2d, w_stop_2d, filter_nums, fs)
                filtered_data = org_filtered_data['bank1']
                # for bank in filtered_data:
                #     raw_data = filtered_data[bank]
                #     tmp_data = cutting_data(raw_data, t_delay, t_reaction, t_task, fs)
                #     filtered_data[bank] = tmp_data
                [nChannels, nTimes, nEvents, nTrials] = data.shape

                tmp_acc = 0

                if method == 'FBCCA':
                    test = FilterBankCCA(data=data, wPass2D=w_pass_2d, wStop2D=w_stop_2d, numFilter=filter_nums,
                                         freqsList=freqsList, nHarmonics=n_harmonics, fs=fs)
                    for trial in range(nTrials):
                        print('-' * 50)
                        print('正在测试第{}个试验数据'.format(trial + 1))
                        test.leave_one(trial)
                        res = test.predict(trial)
                        _, t = acc_compute(res)
                        tmp_acc += t
                    Acc = tmp_acc / nTrials
                    acc_all.append(Acc)
                    itr = cal_itr(nEvents, t_task, Acc)
                    itr_all.append(itr)
                    print('-' * 50)
                    print('准确率为{:05f}'.format(Acc))
                    print('ITR=', itr)

                if method == 'eCCA':
                    test = eCCA(data=filtered_data, nHarmonics=n_harmonics, fs=fs, freqsList=freqsList)
                    for trial in range(nTrials):
                        print('-' * 50)
                        print('正在测试第{}个试验数据'.format(trial + 1))
                        test.leave_one(trial)
                        res = test.predict(trial)
                        _, t = acc_compute(res)
                        tmp_acc += t
                    Acc = tmp_acc / nTrials
                    acc_all.append(Acc)
                    itr = cal_itr(nEvents, t_task, Acc)
                    itr_all.append(itr)
                    print('-' * 50)
                    print('准确率为{:05f}'.format(Acc))
                    print('ITR=', itr)

                if method == 'MsetCCA':
                    test = MutiSetCCA(data=filtered_data, fs=fs)
                    for trial in range(nTrials):
                        print('-' * 50)
                        print('正在测试第{}个试验数据'.format(trial + 1))
                        test.leave_one(trial)
                        test.fit()
                        res = test.predict(trial)
                        _, t = acc_compute(res)
                        tmp_acc += t
                    Acc = tmp_acc / nTrials
                    acc_all.append(Acc)
                    itr = cal_itr(nEvents, t_task, Acc)
                    itr_all.append(itr)
                    print('-' * 50)
                    print('准确率为{:05f}'.format(Acc))
                    print('ITR=', itr)

                if method == 'MsetFB_eCCA':
                    test = MsetFB_eCCA(data=data, wPass2D=w_pass_2d, wStop2D=w_stop_2d, numFilter=filter_nums,
                                       freqsList=freqsList, nHarmonics=n_harmonics, fs=fs)
                    for trial in range(nTrials):
                        print('-' * 50)
                        print('正在测试第{}个试验数据'.format(trial + 1))
                        test.leave_one(trial)
                        test.fit()
                        res = test.predict(trial)
                        _, t = acc_compute(res)
                        tmp_acc += t
                    Acc = tmp_acc / nTrials
                    acc_all.append(Acc)
                    itr = cal_itr(nEvents, t_task, Acc)
                    itr_all.append(itr)
                    print('-' * 50)
                    print('准确率为{:05f}'.format(Acc))
                    print('ITR=', itr)

            print('*' * 50)
            print('+' * 50)
            print('平均准确率为{:05f}'.format(sum(acc_all) / len(acc_all)))
            print('平均ITR=', sum(itr_all) / len(itr_all))
            sio.savemat(save_res_path + '/' + filename,
                        {'acc': acc_all, 'itr': itr_all, 'mean_acc': sum(acc_all) / len(acc_all),
                         'mean_itr': sum(itr_all) / len(itr_all)})
            worksheet1.write_row('A{}'.format(write_point), ['t_task', t_task])
            write_point += 1
            worksheet1.write_row('A{}'.format(write_point),
                                 ['mean_acc', sum(acc_all) / len(acc_all), 'mean_itr', sum(itr_all) / len(itr_all)])
            write_point += 1
            worksheet1.write_row('A{}'.format(write_point), acc_all)
            write_point += 1
            worksheet1.write_row('A{}'.format(write_point), itr_all)
            write_point += 1
            del acc_all, itr_all
    workbook.close()
