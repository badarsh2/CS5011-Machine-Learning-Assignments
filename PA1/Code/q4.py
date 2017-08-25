from numpy import array, c_, r_, float32, mean
import random

def impute(arr):
   """
   Missing values are replaced with median of the input values for that feature
   """
   arr = array(arr)
   cities = arr[:,3]
   datapts = c_[arr[:,:3],arr[:,4:]]

   datapts[datapts=='?'] = '-1'
   datapts_ = array(datapts,dtype=float32)
   temp = datapts_.copy()
   for i in range(temp.shape[1]):
       # Using the sample mean to replace missing values
       temp[:,i][temp[:,i] == -1] = mean(temp[:,i][temp[:,i] > -1])
   temp_filled = array(temp, dtype=str)
   cities = cities.reshape((-1,1))
   data_final = c_[temp_filled[:,:3],cities,temp[:,3:]]

   # First 5 columns are non-predictive
   # Last column is the goal
   data_final_filtered = data_final[:,5:]

   return data_final_filtered.tolist()

def splitdata(data, test_to_train_ratio, num):
   """
   Splits the imputed data into 'num' 80:20 splits
   """
   total = len(data)
   test_count = int(test_to_train_ratio*total)
   arr = []
   data = array(data)
   for i in range(1, num+1):
       idx = range(total)
       random.shuffle(idx)
       test_idx = idx[:test_count]
       train_idx = idx[test_count:]
       arr.append((data[train_idx], data[test_idx]))

   return arr

def main():
	test_to_train_ratio = 0.2
	n_splits = 5
	arr = []


	# Reading the CandC dataset file 
	fin = open("communities.data.txt",'r')
	lines = fin.readlines()
	for l in lines:
		ll = l.strip()
		a = ll.split(',')
		arr.append(a)
	fin.close()

	# Carrying out imputation
	data_list = impute(arr)

	# Turning in the completed dataset
	fout = open("CandC.csv",'w')
	for i in range(len(data_list)):
		fout.write(','.join(data_list[i]))
		fout.write('\n')
	fout.close()
	arr = splitdata(data_list, test_to_train_ratio, n_splits)

	# 80:20 Data splitting for Q5 and Q6
	for i in range(1, n_splits + 1):
		fout = open("CandC-train" + str(i) + ".csv",'w')
		for j in range(len(arr[i-1][0])):
			fout.write(','.join(arr[i-1][0][j]))
			fout.write('\n')
		fout.close()
		fout2 = open("CandC-test" + str(i) + ".csv",'w')
		for j in range(len(arr[i-1][1])):
			fout2.write(','.join(arr[i-1][1][j]))
			fout2.write('\n')
		fout2.close()

if __name__ == '__main__':
	main()