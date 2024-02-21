import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, root_path, train_path, test_path):
        self.root_path = root_path
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        return

    def get_data_shape(self):
        return self.train_data[0].shape

    def get_data_real_shape(self):
        return self.train_data.shape

    def load_train(self):
        data = pd.read_csv(self.root_path + self.train_path)
        labels = np.asarray(data.iloc[:, 0].values)
        data = np.asarray(data.iloc[:, 1:].values)
        return labels, data

    def load_test(self):
        data = pd.read_csv(self.root_path + self.test_path)
        data = np.asarray(data.iloc[:, :].values)
        return data

    def load_data(self):
        self.train_labels, self.train_data = self.load_train()
        self.test_data = self.load_test()
        return

    def reshape_data_to_image_shape(self):
        former_shape = self.get_data_shape()
        sqr_size = int(former_shape[0]**0.5)
        self.train_data = self.train_data.reshape(len(self.train_data), sqr_size, sqr_size, 1)
        self.test_data = self.test_data.reshape(len(self.test_data), sqr_size, sqr_size, 1)
        return

    def my_preprocess(self, data):
        return data/255

    def preprocess_data(self):
        self.train_data = self.my_preprocess(self.train_data)
        #self.train_labels = self.train_labels.reshape(self.train_labels.shape[0], 1)
        self.test_data = self.my_preprocess(self.test_data)
        return

if __name__ == "__main__":
    root = './data/'
    train_data_name = 'train.csv'
    test_data_name = 'test.csv'
    dl = DataLoader(root, train_data_name, test_data_name)
    dl.load_data()
    print(len(dl.train_data))