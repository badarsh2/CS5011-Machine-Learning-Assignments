import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def regcalc(train_set, k = 3):
	"""
	Trains a k-NN classifier based on the training set
	"""
	# Extracting labels
	Y = train_set[:,0]

	# Extracting X
	X = train_set[:,1:]

	# Training a linear regressor
	classifier = KNeighborsClassifier(n_neighbors = k)
	classifier.fit(X, Y)
	return classifier

def precisionrecallcalc(n_classes, prediction, actual):
	"""
	Caiculates precision and recall
	"""
	precision = precision_score(actual, prediction, average="binary")
	recall = recall_score(actual, prediction, average="binary")
	average_precision = precision_score(actual, prediction, average="binary")
	accuracy = accuracy_score(actual, prediction)
	fscore = 2*precision*recall/(precision + recall)
	print "Accuracy:"
	print accuracy
	print "Precision:"
	print precision
	print "Recall:"
	print recall
	print "F-score:"
	print fscore

def main():
	# Reading the data from the datasets created in Q1
	train_set = np.loadtxt('DS1-train.csv', delimiter=',')
	test_set = np.loadtxt('DS1-test.csv', delimiter=',')

	n_samples, n_features = train_set.shape
	n_classes = 2
	model = regcalc(test_set, 10)
	prediction = model.predict(test_set[:,1:])

	precisionrecallcalc(n_classes, prediction, test_set[:,0])


if __name__ == '__main__':
	main()