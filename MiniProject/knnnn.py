# <-- Tianyao.York.Zhang -->
# <-- 2019.8.16 -->
# <-- Mini=Project -->

import scipy.io as scio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt


K_MAX = 30


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


def k_selection(x, y):
	result_mean = []
	result_std = []
	method = ['euclidean', 'manhattan', 'chebyshev']
	count = 0
	mini = 0
	for k in range(1, K_MAX):
		for m in range(len(method)):
			k_model = KNeighborsClassifier(n_neighbors = k, metric = method[m])
			result = cross_val_score(k_model, x, y, cv = 10)
			result_mean.append([k, m, np.mean(result)])
			result_std.append([k, m, np.std(result)])
			if result_mean[mini][2] < result_mean[count][2]:
				mini = count
			count += 1
	print(result_mean, result_std)
	return result_mean[mini][0], method[result_mean[mini][1]]


def run():
	x_train, y_train, x_test, y_test = import_data_set()
	k_neighbors, method = k_selection(x_train, y_train)
	print(k_neighbors)
	model = KNeighborsClassifier(n_neighbors = k_neighbors, metric = method)
	model.fit(x_train, y_train)
	score = model.score(x_test, y_test)
	print(score)


if __name__ == '__main__':
	run()