from abc import abstractmethod, ABCMeta


# abc.ABCMeta 是一个metaclass，用于在Python程序中创建抽象基类；抽象类不能被实例化。
class BasicCCA(metaclass=ABCMeta):
    def __init__(self, data, fs=1000):
        """
        基础CCA类

        -------------
        :param data: 脑电数据，注意输入的格式为[通道x时间点x事件x试验]
        :param fs: 采样频率(float)，单位赫兹，默认是1kHz
        """
        # config model
        self.data = data
        self.fs = fs

    @abstractmethod  # 声明“抽象方法”
    def leave_one(self, selected_trial):
        pass

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test, y_test):
        pass


if __name__ == '__main__':
    pass
