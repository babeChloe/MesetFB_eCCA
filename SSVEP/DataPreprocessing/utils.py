from scipy import signal


class FiltersUtils:
    """
    - 滤波器组分析工具类

    -------------------

    - 引用：无
    - 创建时间: 2023/3/19 22:19
    ------------------

    - 更新日志:
        - 2023/3/19 22:19  Li JiaYi     创建工具类
    """

    def __get_iir_sos_band(self, wPass, wStop, fs):
        """
        获得切比雪夫I型滤波器的二阶成分

        -----------------------
        - 引用：

        -----------------------

        :param wPass: list, 2 elements
        :param wStop: list, 2 elements
        :return: sos_system
            i.e the filter coefficients.
        """
        if len(wPass) != 2 or len(wStop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if wPass[0] > wPass[1] or wStop[0] > wStop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if wPass[0] < wStop[0] or wPass[1] > wStop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

        wp = [2 * wPass[0] / fs, 2 * wPass[1] / fs]
        ws = [2 * wStop[0] / fs, 2 * wStop[1] / fs]
        gpass = 3
        gstop = 40  # dB

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')

        return sos_system

    def ChebyshevI_BandpassFilters(self, data, wPass2D, wStop2D, numFilter, fs):
        """
        切比雪夫I型滤波（采用FBCCA的子带思想）

        引用：
        --------------------------------------------------------

        :param wPass2D: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param wStop2D: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data:4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return:滤波后的数据, dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
        """
        # print("开始切比雪夫带通滤波....")
        if wPass2D.shape != wStop2D.shape:
            raise ValueError('？通带和阻带参数总数目不一致？')
        if numFilter > wPass2D.shape[1]:
            raise ValueError('滤波器数目必须小于等于通阻带参数总数！')

        sosSystem = dict()
        filteredData = dict()
        for idxFilter in range(numFilter):
            sosSystem['filter' + str(idxFilter + 1)] = self.__get_iir_sos_band(
                wPass=[wPass2D[0, idxFilter], wPass2D[1, idxFilter]], wStop=[wStop2D[0, idxFilter],
                                                                             wStop2D[1, idxFilter]], fs=fs)
            filterData = signal.sosfiltfilt(sosSystem['filter' + str(idxFilter + 1)], data, axis=1)
            filteredData['bank' + str(idxFilter + 1)] = filterData
        # print("完成滤波！")
        return filteredData

    def filered_data_notch(self, data, fs, F0=50, Q=35):
        """
        陷波滤波器去除工频干扰

        引用：https://blog.csdn.net/qq_41978536/article/details/90736793

        -------------------------------------------------------------------
        :param data: 输入数据

        :param F0: 需要去除的频率，默认为50Hz
        :param Q: 质量因子
        :return: 滤波后数据
        """
        # print('开始陷波滤波去除工频干扰...')
        w0 = F0 / (fs / 2)
        b, a = signal.iirnotch(w0, Q)
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        # print('完成滤波！')
        return filtered_data


def cutting_data(data, t_delay, t_reaction, t_task, fs):
    beginpoint = int((t_delay + t_reaction) * fs)
    endpoint = int((t_delay + t_reaction + t_task) * fs)
    data = data[:, beginpoint:endpoint, :, :]
    data = data[:, :int(fs * t_task), :, :]
    return data


if __name__ == '__main__':
    pass
