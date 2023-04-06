class Benchmark:
    '''
    引用：Wang Y , Chen X , Gao X , et al. A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces[J].IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017, 25(10):1746-1752.

    ------------------

    该数据集收集了35名健康受试者（17名女性，17-34岁，平均年龄：22岁）的SSVEP-BCI记录，集中在40个以不同频率（8-15.8Hz，间隔0.2Hz）闪烁的字符上。

    对每个受试者来说，实验由6个区块组成。每个区块包含40个试验，对应于以随机顺序显示的所有40个字符。
    每个试验开始时都有一个视觉提示（一个红色的方块），表示一个目标刺激。该提示在屏幕上出现了0.5秒。
    受试者被要求在提示时间内尽快将他们的目光转移到目标上。
    在线索偏移之后，所有的刺激开始在屏幕上同时闪烁，持续5秒。
    在刺激偏移之后，屏幕在下一次试验开始之前空白0.5秒，这使得受试者在连续试验之间有短暂的休息。
    为了便于视觉固定，在刺激期间，闪烁的目标下方出现一个红色的三角形。在每个区块中，要求被试在刺激期间避免眨眼。
    为了避免视觉疲劳，在两个连续的区块之间有几分钟的休息时间。

    脑电图数据是用Synamps2系统（Neuroscan, Inc.）获得的，采样率为1000 Hz。
    放大器的频率通带范围从0.15赫兹到200赫兹。
    六十四个通道覆盖了受试者的整个头皮，并根据国际10-20系统进行了排列。
    地面零电位被放置在Fz和FPz之间的中间位置。参照位于顶点。电极阻抗保持在10KΩ以下。
    为了消除常见的电力线噪音，在数据记录中使用了50赫兹的陷波滤波器。
    由计算机产生的事件触发器送到放大器，并记录在一个与脑电图数据同步的事件通道上。
    连续的脑电图数据被分割成6秒的事件（刺激前500毫秒，刺激后5.5秒开始）。
    随后，这些时间段被降频为250赫兹。因此，每个试验由1500个时间点组成。
    最后，这些数据在MATLAB中被存储为双精度浮点值，并被命名为mat（即S01.mat, ..., S35.mat）。

    对于每个文件，在MATLAB中加载的数据产生一个名为'数据'的四维矩阵，尺寸为[64, 1500, 40, 6]。
    这四个维度表示'电极索引'、'时间点'、'目标索引'和'块索引'。
    电极的位置被保存在一个'64-通道.loc'文件中。
    每个SSVEP频率有六个试验。40个目标指数的频率和相位值被保存在 "Freq_Phase.mat "文件中。

    所有受试者的信息都列在一个'Sub_info.txt'文件中。
    对于每个受试者，有五个因素，包括'受试者指数'、'性别'、'年龄'、'惯用手'和'组别'。
    根据受试者在基于SSVEP的BCI中的经验，他们被分为 "有经验 "组（8名受试者，S01-S08）和 "无经验 "组（27名受试者，S09-S35）。

    Frequency Table
    8    9   10   11   12   13   14   15
    8.2  9.2 10.2 11.2 12.2 13.2 14.2 15.2
    8.4  9.4 10.4 11.4 12.4 13.4 14.4 15.4
    8.6  9.6 10.6 11.6 12.6 13.6 14.6 15.6
    8.8  9.8 10.8 11.8 12.8 13.8 14.8 15.8
    '''

    CHANNELS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
        'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
        'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FC8', 'T7',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
        'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
        'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]

    def get_freqs_and_phases(self, dataset_path):
        """
        获取数据集的频率和相位

        -----------------------------------
        :param dataset_path: 数据集路径
        :return: 数据集的频率和相位
        """
        import scipy.io as sio
        try:
            stim_para = sio.loadmat(dataset_path + '/Freq_Phase.mat')
            # 获取数据集的频率
            self.freqs = stim_para['freqs'].squeeze()
            # 获取数据集的相位
            self.phases = stim_para['phases'].squeeze()
            del stim_para
            return self.freqs, self.phases
        except Exception as e:
            raise "读取数据出错！" + str(e)

    def load_single_subject_data(self, filePath, chans):
        """
        加载单个受试者数据

        -----------------------------------
        :param filePath: 文件路径
        :param chans: 通道list
        :return:原始数据
        """
        import scipy.io as sio
        try:
            raw_mat = sio.loadmat(filePath)
            # 通道 时间片 事件数 试验
            raw_data = raw_mat['data']  # (64, 1500, 40, 6)
            idx_loc = list()
            if isinstance(chans, list):
                for _, char_value in enumerate(chans):
                    idx_loc.append(self.CHANNELS.index(char_value.upper()))
            raw_data = raw_data[idx_loc, :, :, :] if idx_loc else raw_data
            return raw_data
        except Exception as e:
            raise "读取数据出错！" + str(e)

    def load_subjects_data(self, subjectList, filePath, chans):
        """
        加载多组受试者数据

        -----------------------------------
        :param subjectList: 受试者集
        :param filePath: 文件路径
        :param chans: 通道list
        :return: 原始数据集
        """
        print('\n')
        print("/-/"*30)
        try:
            print('正在加载数据....')
            data = dict()
            for subject in subjectList:
                print('加载{}数据....'.format(subject))
                data[subject] = self.load_single_subject_data(filePath+"/"+subject+".mat", chans)
            print('加载完成！')
            print("/-/" * 30)
            return data
        except Exception as e:
            raise "读取数据出错！" + str(e)

    def download_data(self, save_path):
        pass


if __name__ == '__main__':
    pass
