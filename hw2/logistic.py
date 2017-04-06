import numpy as np
import sys
import random

a = list(range(59, 106, 1))
get = list(range(6))

train_data = np.genfromtxt( sys.argv[1], delimiter = ',' )
train_data = np.delete( train_data, [0], 0 )


train_ans = np.genfromtxt( sys.argv[2] )

test_data = np.genfromtxt( sys.argv[3], delimiter = ',' )
test_data = np.delete( test_data, [0], 0 )


for i in range( int(len(train_data)) ):
	if train_data[i][14] == 1:		train_data[i][9] = 1
	if train_data[i][52] == 1:		train_data[i][47] = 1
	if train_data[i][105] == 1:		train_data[i][102] = 1

for i in range( int(len(test_data)) ):
	if test_data[i][14] == 1:		test_data[i][9] = 1
	if test_data[i][52] == 1:		test_data[i][47] = 1
	if test_data[i][105] == 1:		test_data[i][102] = 1


mu = np.mean(train_data, axis = 0)
sd = np.std(train_data, axis = 0)
train_data = (train_data - mu) / sd
test_data = (test_data - mu) / sd

train_data = np.delete( train_data, a, 1 )
test_data = np.delete( test_data, a, 1 )

train_data_bp = np.concatenate( (train_data, train_data[:, get ]**2), axis = 1 )
train_data_bp = np.concatenate( (train_data_bp, np.exp(train_data[:, get ]) ), axis = 1 )

test_data_bp = np.concatenate( (test_data, test_data[:, get ]**2), axis = 1 )
test_data_bp = np.concatenate( (test_data_bp, np.exp(test_data[:, get ]) ), axis = 1 )



train_data = train_data_bp


num_features = train_data.shape[1]
it = 9000

b = 0.0
w = np.zeros( num_features )


lr = 1.25e-2
b_lr = 0.0
w_lr = np.zeros( num_features )
lamda = 0

for i in range(it):

	b_grad = 0.0
	w_grad = np.zeros( num_features )

	

	z = np.dot(train_data, w) + b
	fx = 1 / ( 1 + np.exp(-z) )				# sigmoid


	b_grad = -np.sum(train_ans - fx)
	w_grad = -np.dot( (train_ans - fx), train_data ) - 2 * lamda * w

	
	b_lr += b_grad**2;		b -= lr/np.sqrt(b_lr) * b_grad
	w_lr += w_grad**2;		w -= lr/np.sqrt(w_lr) * w_grad




test_data = test_data_bp


z = np.dot(test_data, w) + b
ans = 1 / (1 + np.exp(-z))



f = open( sys.argv[4], 'w')
print("id,label", file = f)
for i in range( int(len(ans)) ):
	if ans[i] > 0.5:
		print(i+1, '1', sep = ',', file = f)
	else:
		print(i+1, '0', sep = ',', file = f)
