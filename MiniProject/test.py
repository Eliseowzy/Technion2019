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


def run():
	x_train, y_train, x_test, y_test = import_data_set()
	res = []
	for k in range(1, K_MAX):
		model = KNeighborsClassifier(n_neighbors = k)
		model.fit(x_train, y_train)
		score = model.score(x_test, y_test)
		res.append(score)
	print(res)
	print(max(res))
	print(res.index(max(res)))
	

run()