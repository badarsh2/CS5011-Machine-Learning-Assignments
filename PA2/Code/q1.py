"""
Program to demonstrate SVM using various kernels
"""

from numpy import genfromtxt, ascontiguousarray, sum, mean, linspace, logspace, zeros, object
from random import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from scipy.io import savemat

# Linear Kernel
def linearkerneleval(train_X, train_Y, k_fold, obj_arr_1):
	# r = []
	best_c = 0
	best_acc = 0
	for c in logspace(-4, 4, 10):
	    acc = []
	    # Performing K-fold cross validation test
	    for train, test in k_fold:
	        model = OneVsRestClassifier(svm.SVC(kernel="linear", C=c, max_iter=100000))
	        model.fit(train_X[train], train_Y[train])
	        pred = model.predict(train_X[test])
	        acc.append(100.*sum(pred==train_Y[test])/pred.shape[0])
	    # Choosing the model with best accuracy
	    if(mean(acc) > best_acc):
	    	best_acc, best_c = mean(acc), c
	    print 'C =',c,'Avg.acc =',mean(acc)
	obj_arr_1[0][0] = 'C'
	obj_arr_1[1][0] = 'Accuracy'
	obj_arr_1[0][1] = best_c
	obj_arr_1[1][1] = best_acc

# Polynomial Kernel
def polykerneleval(train_X, train_Y, k_fold, obj_arr_2):
	r = []
	best_coef0 = 0
	best_degree = 0
	best_c = 0
	best_acc = 0
	for coef0 in linspace(-0.5, 0.5, 11):
		for c in logspace(-4, 4, 10):
			for degree in range(2, 6):
			    acc = []
			    # Performing K-fold cross validation test
			    for train, test in k_fold:
			        model = OneVsRestClassifier(svm.SVC(kernel="poly", C=c, coef0=coef0, degree=degree))
			        model.fit(train_X[train], train_Y[train])
			        pred = model.predict(train_X[test])
			        acc.append(100.*sum(pred==train_Y[test])/pred.shape[0])
			    # Choosing the model with best accuracy
			    if(mean(acc) > best_acc):
			    	best_acc, best_c, best_degree, best_coef0 = mean(acc), c, degree, coef0
			    print 'coef0',coef0,'degree =',degree,'C =',c,'Avg.acc =',mean(acc)
	obj_arr_2[0][0] = 'C'
	obj_arr_2[1][0] = 'Coef0'
	obj_arr_2[2][0] = 'Degree'
	obj_arr_2[3][0] = 'Accuracy'
	obj_arr_2[0][1] = best_c
	obj_arr_2[1][1] = best_coef0
	obj_arr_2[2][1] = best_degree
	obj_arr_2[3][1] = best_acc

# RBF Kernel
def rbfkerneleval(train_X, train_Y, k_fold, obj_arr_3):
	r = []
	best_gamma = 0
	best_c = 0
	best_acc = 0
	for c in logspace(-4, 4, 10):
		for gamma in logspace(-4, 4, 10):
			acc = []
		    # Performing K-fold cross validation test
			for train, test in k_fold:
				model = OneVsRestClassifier(svm.SVC(kernel="rbf", C=c, gamma=gamma))
				model.fit(train_X[train], train_Y[train])
				pred = model.predict(train_X[test])
				acc.append(100.*sum(pred==train_Y[test])/pred.shape[0])
			# Choosing the model with best accuracy
			if(mean(acc) > best_acc):
			    	best_acc, best_gamma, best_c = mean(acc), gamma, c
			print 'C =',c, 'gamma = ', gamma, 'Avg.acc =',mean(acc)
	obj_arr_3[0][0] = 'C'
	obj_arr_3[1][0] = 'Gamma'
	obj_arr_3[2][0] = 'Accuracy'
	obj_arr_3[0][1] = best_c
	obj_arr_3[1][1] = best_gamma
	obj_arr_3[2][1] = best_acc
	print "Best parameters for RBF kernel:"
	print "C: ", c, "gamma", best_gamma, "Accuracy:", best_acc

# Sigmoidal Kernel
def sigmoidkerneleval(train_X, train_Y, k_fold, obj_arr_4):
	r = []
	best_coef0 = 0
	best_c = 0
	best_acc = 0
	for c in logspace(-4, 4, 10):
		for coef0 in linspace(-0.5, 0.5, 11):
			acc = []
		    # Performing K-fold cross validation test
			for train, test in k_fold:
				model = OneVsRestClassifier(svm.SVC(kernel="sigmoid", C=c, coef0=coef0))
				model.fit(train_X[train], train_Y[train])
				pred = model.predict(train_X[test])
				acc.append(100.*sum(pred==train_Y[test])/pred.shape[0])
			# Choosing the model with best accuracy
			if(mean(acc) > best_acc):
			    	best_acc, best_coef0, best_c = mean(acc), coef0, c
			print 'C =',c, 'coef0 = ', coef0, 'Avg.acc =',mean(acc)
	obj_arr_4[0][0] = 'C'
	obj_arr_4[1][0] = 'Coef0'
	obj_arr_4[2][0] = 'Accuracy'
	obj_arr_4[0][1] = best_c
	obj_arr_4[1][1] = best_coef0
	obj_arr_4[2][1] = best_acc

def main():
	train_data = genfromtxt(fname = "../Dataset/DS2_train.csv", delimiter=",")
	test_data = genfromtxt(fname = "../Dataset/DS2_test.csv", delimiter=",")
	shuffle(train_data)

	# Transformation to improve performance
	scaler = StandardScaler()
	train_X = scaler.fit_transform(train_data[:,:-1])
	train_Y = ascontiguousarray(train_data[:, -1])

	test_X = scaler.transform(test_data[:,:-1])
	test_Y = ascontiguousarray(test_data[:, -1])

	# 5-fold cross validation
	k_fold = KFold(len(train_X), n_folds = 5)

	# Objects for storing the best parameters and writing to mat file
	obj_arr_1 = zeros((2,2), dtype=object)
	obj_arr_2 = zeros((4,2), dtype=object)
	obj_arr_3 = zeros((3,2), dtype=object)
	obj_arr_4 = zeros((3,2), dtype=object)
	
	print "\nLinear kernel:"
	linearkerneleval(train_X, train_Y, k_fold, obj_arr_1)
	print "\nPoly kernel:"
	polykerneleval(train_X, train_Y, k_fold, obj_arr_2)
	print "\nRBF kernel:"
	rbfkerneleval(train_X, train_Y, k_fold, obj_arr_3)
	print "\nSigmoid kernel:"
	sigmoidkerneleval(train_X, train_Y, k_fold, obj_arr_4)

	savemat('MM14B001.mat', mdict={'Model1': obj_arr_1, 'Model2': obj_arr_2, 'Model3': obj_arr_3, 'Model4': obj_arr_4})

if __name__ == '__main__':
    main()