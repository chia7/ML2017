import numpy as np
import sys


train_data = np.genfromtxt( sys.argv[1], delimiter = ',' )
train_data = np.delete( train_data, [0], 0 )
train_data = train_data.T


train_ans = np.genfromtxt( sys.argv[2] )

valid = (train_ans == 1)
class_1 = train_data[:, valid].T
num_1 = class_1.shape[0]

valid = (train_ans == 0)
class_2 = train_data[:, valid].T
num_2 = class_2.shape[0]


mu_1 = np.mean(class_1, axis = 0)
mu_2 = np.mean(class_2, axis = 0)



sigma_1 = np.zeros(106*106)
sigma_1 = sigma_1.reshape( (106, 106) )
for i in range(num_1):
	sigma_1 += np.dot( (class_1[i] - mu_1).reshape((106,1)), (class_1[i] - mu_1).reshape((1,106)) )
sigma_1 = sigma_1 / num_1


sigma_2 = np.zeros(106*106)
sigma_2 = sigma_2.reshape( (106, 106) )
for i in range(num_2):
	sigma_2 += np.dot( (class_2[i] - mu_2).reshape((106,1)), (class_2[i] - mu_2).reshape((1,106)) )
sigma_2 = sigma_2 / num_2



sigma = (num_1 * sigma_1 + num_2 * sigma_2) / (num_1 + num_2)
w = np.dot( (mu_1 - mu_2).T, np.linalg.inv(sigma) ).T
b = -(1/2) * np.dot( np.dot( mu_1.T, np.linalg.inv(sigma) ), mu_1 ) + (1/2) * np.dot( np.dot( mu_2.T, np.linalg.inv(sigma) ), mu_2 ) + np.log(num_1/num_2)




test_data = np.genfromtxt( sys.argv[3], delimiter = ',' )
test_data = np.delete( test_data, [0], 0 )



z = np.dot(test_data, w) + b
ans = 1 / (1 + np.exp(-z))


f = open( sys.argv[4], 'w')
print("id,label", file = f)
for i in range( int(len(ans)) ):
	if ans[i] > 0.5:
		print(i+1, ',1', sep = '', file = f)
	else:
		print(i+1, ',0', sep = '', file = f)
