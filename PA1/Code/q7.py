from numpy import count_nonzero, array, unique, linspace, mean, c_, r_
from sklearn import linear_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from linear_classification import linear_classifier_learn, report_accuracy

def measure_f_score(precision, recall):
   num = 2*precision*recall
   den = precision + recall
   # Handling division by zero case, which is caused by precision = recall = 0
   if den == 0:
       return 0
   return num/den

def measure_accuracy(pred, exact):
   misclassified = count_nonzero(pred.ravel() - exact.ravel())
   total = len(pred.ravel())
   accuracy = 1.0*(total-misclassified)/total
   return accuracy

def measure_precision(pred, exact, label = 1):
   _pred = pred.ravel()
   _exact = exact.ravel()
   pred_pos = count_nonzero(_pred == label)
   pred_pos_true = count_nonzero((_pred == label) & (_exact == label))
   return 1.0*pred_pos_true/pred_pos

def measure_recall(pred, exact, label = 1):
   _pred = pred.ravel()
   _exact = exact.ravel()
   exact_pos = count_nonzero(_exact == label)
   pred_pos_true = count_nonzero((_pred == label) & (_exact == label))
   return 1.0*pred_pos_true/exact_pos

def report_accuracy(test_set, model, thresh = 0.5, output1 = 0.0, output2 = 1.0, label = 1):
   # Extracting X
   X = test_set[:,:-1]

   # Extracting labels
   Y = test_set[:,-1]

   # Predicted labels
   pred = model.predict(X)
   pred[pred <= thresh] = output1
   pred[pred > thresh] = output2

   accuracy = measure_accuracy(pred, Y)
   precision = measure_precision(pred, Y, label)
   recall = measure_recall(pred, Y, label)
   f_score = measure_f_score(precision, recall)

   return accuracy, precision, recall, f_score

def plot(X, X_ex, Y, model_reduced, title):
   """
   A utility function to plot the given 3-D dataset and the reduced dataset obtained by extracting features ( a single feature )
   """
   fig = plt.figure('Raw data', figsize = (4, 3))
   plt.clf()
   ax = Axes3D(fig, rect = [0, 0, .95, 1], elev = 48, azim = 134)
   ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = Y, cmap = plt.cm.binary_r, label = ['Class-1', 'Class-2'])
   x_surf = [X[:, 0].min(), X[:, 0].max(), X[:, 0].min(), X[:, 0].max()]
   y_surf = [X[:, 0].max(), X[:, 0].max(), X[:, 0].min(), X[:, 0].min()]
   x_surf = array(x_surf)
   y_surf = array(y_surf)

   ax.w_xaxis.set_ticklabels([])
   ax.w_yaxis.set_ticklabels([])
   ax.w_zaxis.set_ticklabels([])

   fig2 = plt.figure('Projected data')
   labels = unique(Y.tolist())
   idx_ro = (Y == labels[0])
   idx_go = (Y == labels[1])
   p1 = plt.plot(X_ex[idx_ro], Y[idx_ro], 'ko')
   p2 = plt.plot(X_ex[idx_go], Y[idx_go], 'wo')

   c = 0.5*(labels[0] + labels[1])
   m = model_reduced.coef_[0]
   x_val = linspace(X_ex.min(), X_ex.max(), 11)
   y_val = c + m*x_val
   p3 = plt.plot(x_val, y_val, color = 'blue', linewidth = 2)
   plt.legend(['Class-1', 'Class-2', 'Classifier boundary'])
   plt.xlim(X_ex.min()*0.9, X_ex.max()*1.1)
   plt.ylim(Y.min()*0.9, Y.max()*1.1)
   plt.show()

def linear_classifier_learn(train_set):
   # Extracting X
   X = train_set[:,:-1]

   # Extracting labels
   Y = train_set[:,-1]

   # Training a linear regressor
   regr = linear_model.LinearRegression()
   regr.fit(X, Y)

   return regr

def fit_pca(X, n_components):
   pca = PCA(n_components = n_components)
   pca.fit(X)
   return pca

def extract_features_pca(pca, X):
   _X = pca.transform(X)
   return _X

def main():
   train_x, train_y, test_x, test_y = [], [], [], []
   trainstream = open("datasets/DS3/train.csv", 'r').readlines()
   trainlabelsstream = open("datasets/DS3/train_labels.csv", 'r').readlines()
   teststream = open("datasets/DS3/test.csv", 'r').readlines()
   testlabelsstream = open("datasets/DS3/test_labels.csv", 'r').readlines()
   for l in trainstream:
       train_x.append(map(float, l.split(',')))
   train_y = map(lambda x: float(x.strip()), trainlabelsstream)   
   thresh = mean(list(set(train_y)))
   train_y = array(train_y).reshape((-1, 1))
   for l in teststream:
       test_x.append(map(float, l.split(',')))
   test_y = map(lambda x: float(x.strip()), testlabelsstream)
   test_y = array(test_y).reshape((-1, 1))

   model_pca = fit_pca(train_x, 1)
   train_x_ex = extract_features_pca(model_pca, train_x)
   test_x_ex = extract_features_pca(model_pca, test_x)

   train_set = c_[train_x_ex, train_y]
   test_set = c_[test_x_ex, test_y]
   data_set = r_[c_[train_x, train_y], c_[test_x, test_y]]

   model = linear_classifier_learn(train_set)

   accuracy, precision, recall, f_score = [[-1 for _ in range(2)] for __ in range(4)]

   # Class-1 params
   accuracy[0], precision[0], recall[0], f_score[0] =  report_accuracy(test_set, model, thresh, 1.0, 2.0, 1.0)
   # Class-2 params
   accuracy[1], precision[1], recall[1], f_score[1] =  report_accuracy(test_set, model, thresh, 1.0, 2.0, 2.0)

   print "Coeff:", model.coef_
   print "Class-1:", accuracy[0], precision[0], recall[0], f_score[0]
   print "Class-2:", accuracy[1], precision[1], recall[1], f_score[1]

   plot(data_set[:,:-1], extract_features_pca(model_pca, data_set[:,:-1]), data_set[:, -1], model, "Plot result")

if __name__ == '__main__':
   main()