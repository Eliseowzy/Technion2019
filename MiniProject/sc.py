# <-- Tianyao.York.Zhang -->
# <-- 2019.8.16 -->
# <-- Mini=Project -->

import scipy.io as scio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def import_data_set():
	data_file = './data/data.mat'
	database = scio.loadmat(data_file)
#	print(database)
	train_set_x = database['trainSet'][0][0][0]
	train_set_y = database['trainSet'][0][0][1]
	test_set_x = database['testSet'][0][0][0]
	test_set_y = database['testSet'][0][0][1]
	train_set_x = train_set_x.T
	train_set_y = train_set_y.T
	test_set_x = test_set_x.T
	test_set_y = test_set_y.T
#	print(train_set_x, train_set_y, test_set_x, test_set_y)
	return train_set_x, train_set_y, test_set_x, test_set_y


def run():
	x_train, y_train, x_test, y_test = import_data_set()
	sc = StandardScaler()
	sc.fit(x_train, y_train)
	x_train_std = sc.transform(x_train)
	x_test_std = sc.transform(x_test)
	
	ppn = Perceptron(eta0 = 0.1, random_state = 0)
	ppn.fit(x_train_std, y_train)
	
	predict = ppn.predict(x_test_std)
	count = 0
	for left, right in zip(predict, y_test):
		if left == right:
			count += 1
	print(count / len(y_test))


if __name__ == '__main__':
	run()