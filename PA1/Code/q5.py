# import numpy as np
from numpy import sum, loadtxt, argmin, uint, float, savetxt
from sklearn import linear_model

def regcalc(train_set):
  """
  Trains a linear regressor based on the training set
  """
  # Extracting X
  X = train_set[:,:-1]

  # Extracting labels
  Y = train_set[:,-1]

  regr = linear_model.LinearRegression()
  regr.fit(X, Y)

  return regr

def reserrorcalc(test_set, model):
  """
  Calculates RSS error for the given test set
  """
  # Extracting X
  X = test_set[:,:-1]

  # Extracting labels
  Y = test_set[:,-1]
  residual_err = sum((model.predict(X) - Y) ** 2)
  return residual_err

def computelinreg(num):

  # array to store the coefficient and residual error from each set
  params = []

  for i in range(1, num+1):
    train_set = loadtxt("CandC-train" + str(i) + ".csv", delimiter=',')
    test_set = loadtxt("CandC-test" + str(i) + ".csv", delimiter=',')
    model = regcalc(train_set)
    residual_error = reserrorcalc(test_set, model)
    params.append((model.coef_, residual_error))
    print "Coefficients learnt in split class " + str(i) + ":"
    print params[i-1][0]
    savetxt("CandC-coeff" +str(i)+".csv", params[i-1][0], delimiter=',')
    print "Residual error of split class " + str(i) + ":"
    print residual_error

  # Least residual error and index
  best_fit_idx = argmin(map(lambda x: x[1], params))
  best_fit_coeff = params[best_fit_idx][0]
  best_residual_err = params[best_fit_idx][1]
  print "Best fit obtained at index:"
  print uint(best_fit_idx)
  print "Residual error of best fit is:"
  print float(best_residual_err)

def main():
  no_of_splits = 5
  computelinreg(no_of_splits)


if __name__ == '__main__':
  main()