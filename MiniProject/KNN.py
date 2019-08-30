import scipy.io as scio
import numpy as np
import operator
import os
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import KFold


def import_data_set():
	data_file = './data/data.mat'
	database = scio.loadmat(data_file)
	logging.info(database)
	train_set_x = database['trainSet'][0][0][0]
	train_set_y = database['trainSet'][0][0][1][0]
	test_set_x = database['testSet'][0][0][0]
	test_set_y = database['testSet'][0][0][1][0]
	logging.info(train_set_x, train_set_y, test_set_x, test_set_y)
	return train_set_x, train_set_y, test_set_x, test_set_y


def auto_norm(data_set):
	min_vals = data_set.min(0)
	max_vals = data_set.max(0)
	ranges = max_vals - min_vals
	norm_data_set = np.zeros(np.shape(data_set))
	m = data_set.shape[0]
	norm_data_set = data_set - np.tile(min_vals, (m, 1))
	norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
	return norm_data_set


def classify(in_x, data_set, labels, k_neighbors):
	in_x = auto_norm(in_x)
	data_set_size = data_set.shape[0]
	logging.info('dataSetSize: %d' % data_set_size)
	print(type(in_x), type(data_set))
	distance = np.tile(in_x, (data_set_size, 1)) - data_set
	distance = distance ** 2
	distance = distance.sum(axis = 1)
	distance = distance ** 0.5
	logging.info(distance)
	sorted_dis = distance.argsort()
	class_count = {}
	for i in range(k_neighbors):
		vote_label = labels[sorted_dis[i]]
		class_count[vote_label] = class_count.get(vote_label, 0) + 1
		sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
	return sorted_class_count[0][0]


def select_k(data, labels):
	range_k = len(data[0])
	data_num = len(data)
	k_mean = np.zeros(range_k)
	k_std = np.zeros(range_k)
	for k in range(range_k):
		
		error_count = []
		
		kf = KFold(n_splits = 2)
		for train_
		for fold in range(9):
			tmp_data = np.delete(data, [int(fold * data_num / 10) ,int((fold + 1) * data_num / 10)], axis = 0)
			tmp_label = np.delete(labels, [int(fold * data_num / 10) ,int((fold + 1) * data_num / 10)], axis = 0)
			data_list = data.tolist()
			test_data = data_list[int(fold * data_num / 10) : int((fold + 1) * data_num / 10)]
			test_data = np.array(test_data)
			data_label = labels.tolist()
			test_label = data_label[int(fold * data_num / 10) : int((fold + 1) * data_num / 10)]
			test_label = np.array(test_label)
			
			tmp_error_count = 0
			for i in range(int(data_num / 10)):
				classifier_result = classify(test_data[i], tmp_data, tmp_label, k)
				logging('the classifier came back with: %d, the real answer is: %d' % (classifier_result, test_label))
				tmp_error_count += classifier_result != test_label[i]
			
			error_count.append(tmp_error_count)
		
		k_mean[k] = np.mean(error_count)
		k_std[k] = np.std(error_count, ddof = 1)
	
	plt.bar(range(range_k), k_mean)
	plt.show()
	plt.bar(range(range_k), k_std)
	plt.show()
	
	return k_mean.index(min(k_mean))


def run():
	train_data, train_label, test_data, test_label = import_data_set()
	k = select_k(train_data, train_label)
	test_num = len(test_data)
	error_count = 0
	for i in range(test_num):
		result = classify(test_data[i], train_data, train_label, k)
		logging('the classifier came back with: %d, the real answer is: %d' % (result, test_label))
		error_count += result != test_label[i]
	print("the total error rate is: %f" % (error_count / test_num))


if __name__ == '__main__':
	run()