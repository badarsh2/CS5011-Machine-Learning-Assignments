from numpy import genfromtxt, min, max as maxx, divide, shape, tile, exp, multiply, unique, sqrt, sum, exp, dot, uint8, argmax, logspace
from numpy.random import randn, random
from numpy.linalg import norm
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support

def sigmoid(x):
    return 1. / (1 + exp(-x))

def sigmoid_derivative(x):
    return multiply(sigmoid(x), 1 - sigmoid(x))

class NeuralNetwork:
	def __init__(self, ni, nh, no):
		# number of input, hidden, and output nodes
		self.ni = ni + 1 # +1 for bias node
		self.nh = nh
		self.no = no

		# Random initialization of the model
		self.w1 = randn(ni, nh) / sqrt(ni)
		# W1 = 2*np.random.random((self.config.nn_input_dim, nn_hdim))-1
		self.b1 = random((1, nh))
		self.w2 = randn(nh, no) / sqrt(nh)
		# W2 = 2*np.random.random((nn_hdim, self.config.nn_output_dim))
		self.b2 = random((1, no))

	def predict(self, X):
	    """
	    Function to predict the label given the input.
	    """
	    w1 = self.w1
	    b1 = self.b1
	    w2 = self.w2
	    b2 = self.b2

	    z1 = X.dot(w1) + b1
	    a1 = sigmoid(z1)
	    z2 = a1.dot(w2) + b2

	    # Applying softmax
	    exp_scores = exp(z2)
	    probs = exp_scores / sum(exp_scores, axis=1, keepdims=True)

	    # Computing the labels
	    labels = argmax(probs, axis=1)
	    return labels

	def cost(self, X, Y):
	    """
	    Function returns the cost of prediction, given the predicted and the actual values.
	    """
	    P = self.predict(X)
	    d = P - Y
	    cost = 0.5*norm(d)
	    return cost

	def accuracy(self, X, Y):
	    """
	    Utility function to return the accuracy given the input, output.
	    """
	    # Obtaining the predicted labels
	    P = self.predict(X)
	    return (100.*sum(P == Y)/Y.shape[0])

	def train_nn(self, X, Y, n_iter, alpha, lambdaa):
	    """
	    Training using a Gradient Descent approach and backpropagating
	    """

	    w1 = self.w1
	    b1 = self.b1
	    w2 = self.w2
	    b2 = self.b2

	    # Training for the given no. of iterations ( n_iter )
	    for i in range(n_iter):
	        z1 = X.dot(w1) + b1
	        a1 = sigmoid(z1)
	        z2 = a1.dot(w2) + b2
	        exp_scores = exp(z2)
	        probs = exp_scores / sum(exp_scores, axis=1, keepdims=True)

	        n_samples = len(X)

	        delta3 = probs
	        # Accounting for softmax layer
	        delta3[range(n_samples), Y] -= 1
	        # Backpropagation
	        dw2 = (a1.T).dot(delta3)
	        db2 = sum(delta3, axis=0, keepdims=True)
	        delta2 = delta3.dot(w2.T) * a1 * (1-a1)
	        dw1 = dot(X.T, delta2)
	        db1 = sum(delta2, axis=0)

	        # Add regularization terms (b1 and b2 don't have regularization terms)
	        # dw2 += lambdaa * w2
	        # dw1 += lambdaa * w1

	        # Gradient descent parameter update
	        w1 += -alpha * dw1
	        b1 += -alpha * db1
	        w2 += -alpha * dw2
	        b2 += -alpha * db2

	        # self.model = { 'w1': w1, 'b1': b1, 'W2': w2, 'b2': b2 }
	        self.w1 = w1
	        self.b1 = b1
	        self.w2 = w2
	        self.b2 = b2

	        if not i % 3000 and i:
	            alpha *= 0.5
	            print 'Alpha changed:', alpha

	        if not i % 100:
	            print 'Iter:', i, 'Cost:', self.cost(X, Y)#, 'Acc:', self.accuracy(test_X, test_Y_labels)

def transform(x, min_val=None, ranges=None):
    if min_val is None:
        min_val = min(x, axis=0)
    if ranges is None:
        ranges = maxx(x - tile(min_val, (shape(x)[0], 1)), axis=0)
    scaled_data = divide((x - tile(min_val, (shape(x)[0], 1))) - 0.0, tile(ranges, (shape(x)[0], 1)))
    return scaled_data, min_val, ranges

def main():
	train_data = genfromtxt(fname = "../Dataset/DS2_train.csv", delimiter=",")
	test_data = genfromtxt(fname = "../Dataset/DS2_test.csv", delimiter=",")
	shuffle(train_data)
	n_classes = len(unique(train_data[:, -1]))

	nn_input = 96  # input layer dimensionality
	nn_output= 4  # output layer dimensionality
	nn_hidden = 40 # hidden layer dimensionality
	alpha = 1e-2

	(train_X, min_train, range_train) = transform(train_data[:, :-1])
	train_Y = uint8(train_data[:, -1])
	(test_X, dum1, dum2) = transform(test_data[:, :-1], min_train, range_train)
	test_Y = uint8(test_data[:, -1])

	# print shape(test_data[:, :-1])
	# print shape(test_X)

	lambda_range = logspace(-2, 2, 5)
	res = []

	for lambdaa in lambda_range:
		network = NeuralNetwork(nn_input, nn_hidden, nn_output)
		network.train_nn(train_X, train_Y, 10000, alpha, lambdaa)
		pred = network.predict(test_X)
		accuracy = network.accuracy(test_X, test_Y)
		precision, recall, f_score, _ = precision_recall_fscore_support(pred, test_Y, average=None)
		res.append((alpha, lambdaa, accuracy, precision, recall, f_score))

	best_fit = max(res, key=(lambda x: x[2]))

	# Displaying performance
	print 'Best model predictions for alpha =', alpha, ' and lambda =', lambdaa
	print 'Acc. =', best_fit[2]
	print 'Precision =', best_fit[3]
	print 'Recall =', best_fit[4]
	print 'F-score =', best_fit[5]

if __name__ == '__main__':
    main()