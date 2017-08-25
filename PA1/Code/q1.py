from numpy import random, c_, r_, zeros, ones, diag, savetxt
from sklearn.cross_validation import train_test_split


nclasses = 2
n_features = 20
n_samples = 2000
test_ratio = 0.3

def generate_data():
	"""
	Generates two-class normally distributed data having same covariance matrices
	"""
	# Two means chosen based on hit-and-trial
	random.seed(0)
	meanmean = random.rand(20,)
	mean1 = meanmean + 0.04
	mean2 = meanmean - 0.04

	temp = random.rand(20,)
	cov = diag(temp)

	x = random.multivariate_normal(mean1, cov, n_samples)
	y = random.multivariate_normal(mean2, cov, n_samples)

	xlabel = zeros(n_samples)
	ylabel = ones(n_samples)

	pts_x = c_[xlabel.reshape((-1,1)), x]
	pts_y = c_[ylabel.reshape((-1,1)), y]


	serial = range(n_samples)
	random.shuffle(serial)
	lim = int(n_samples*test_ratio)
	testserial = serial[:lim]
	trainserial = serial[lim:]

	X_train, X_test, y_train, y_test = train_test_split(pts_x, pts_y, test_size=test_ratio, random_state=42)

	test_set = r_[X_test, y_test]
	train_set = r_[X_train, y_train]

	return test_set, train_set

def write_sets(qno, test_set, train_set):
	savetxt(qno + '-test.csv', test_set, delimiter=',')
	savetxt(qno + '-train.csv', train_set, delimiter=',')

def main():
	test_set, train_set = generate_data()
	write_sets("DS1", test_set, train_set)


if __name__ == '__main__':
	main()
