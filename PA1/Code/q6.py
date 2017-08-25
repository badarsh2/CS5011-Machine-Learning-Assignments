from numpy import loadtxt, sum, c_, argmin, logspace, mean
from sklearn import linear_model

def regcalc(train_set, lambda_ridge):
  """
  Trains a ridge regressor based on the training set and lambda
  """
  # Extracting X
  X = train_set[:,:-1]

  # Extracting labels
  Y = train_set[:,-1]

  regr = linear_model.Ridge(alpha = lambda_ridge)
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

def computeridge(num, lambda_trials, filter_idx = None):

  # Array to store best fit coefficient and residual error for each lambda
  params = []

  for v in lambda_trials:
    # Array to store coefficient and residual error for each lambda and each class
    params_trials = []

    for i in range(1, num+1):
      train_set = loadtxt("CandC-train" + str(i) + ".csv", delimiter=',')
      test_set = loadtxt("CandC-test" + str(i) + ".csv", delimiter=',')
      model = regcalc(train_set, v)
      residual_err = reserrorcalc(test_set, model)
      params_trials.append((model.coef_, residual_err))

    avg_residual_err = mean(map(lambda x: x[1], params_trials))
    best_fit_idx = argmin(map(lambda x: x[1], params_trials))
    best_fit_coeff = params_trials[best_fit_idx][0]

    params.append((best_fit_coeff, avg_residual_err))

  for i in range(len(lambda_trials)):
    print "Lambda = " + str(lambda_trials[i]) + "  Residual error = " + str(params[i][1])

  # FInding best fit among best fits for each lambda
  best_fit_idx = argmin(map(lambda x: x[1], params))
  best_fit_coeff = params[best_fit_idx][0]
  avg_residual_err = params[best_fit_idx][1]
  best_fit_lambda = lambda_trials[best_fit_idx]

  print "\nBest fit obtained for lambda:", best_fit_lambda
  print "Found at index:", best_fit_idx
  print "Residual err:", avg_residual_err

def main():
  no_of_splits = 5

  # Presetting values for lambda
  lambda_trials = logspace(1, 7, 14)*1e-4

  computeridge(no_of_splits, lambda_trials)


if __name__ == '__main__':
  main()