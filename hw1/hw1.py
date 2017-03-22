import sys
import numpy as np
import random

AMB_TEMP = 0;	CH4 = 1;	CO = 2;			NMHC = 3;			NO = 4;				NO2 = 5
NOx = 6;		O3 = 7;		PM10 = 8;		PM25 = 9;			RAINFALL = 10;		RH = 11;
SO2 = 12;		THC = 13;	WD_HR = 14;		WIND_DIREC = 15;	WIND_SPEED = 16;	WS_HR = 17;

train_feature = np.array( [ [PM25, 1] ] )
num_feature = int( np.sum( train_feature[:, 1] ) )

it = 20000
lr = 1.25e-1


def read_train_data(train_feature, num_feature):
	raw_data = np.genfromtxt( sys.argv[1], delimiter = ',' )
	raw_data = np.delete( raw_data, [0], 0 )
	raw_data = np.delete( raw_data, [0, 1, 2], 1 )
	raw_data = np.nan_to_num( raw_data )

	######################################################################
	### get pm2.5 (actual value)

	pm25 = np.array([])			# shape = (1, 5652)
	for month in range(12):
		for day in range(20):
			if day == 0:
				pm25 = np.append( pm25, raw_data[ month*360 + 9, 9: ] )
			else:
				pm25 = np.append( pm25, raw_data[ month*360 + day*18 + 9] )

	######################################################################
	### get selected feature (from the train_feature list)

	feature = np.array([])
	normalize = np.array([])
	for i in range( int( len(train_feature) ) ):
		tmp = np.array([])
		for day in range(240):
			tmp = np.append( tmp, raw_data[ day*18 + train_feature[i][0] ] )

		feature = np.append( feature, tmp )


	feature = feature.reshape( ( num_feature, 5760) )

	######################################################################
	### get all training data ready

	out = np.array([])
	for month in range(12):
		for hour in range(471):
			out = np.append( out, pm25[ month*471 + hour ] )
			for num in range(num_feature):
				out = np.append( out, feature[ num, month*480 + hour : month*480 + hour + 9] )

	out = out.reshape( ( 5652, num_feature*9 + 1 ) )


	return ( out )

def predit(train_feature, num_feature, b, w):
	test = np.genfromtxt( sys.argv[2], delimiter = ',' )
	test = np.nan_to_num( test )
	test = np.delete( test, [0, 1], 1 )

	feature = np.array([])
	for num in range(240):
		tmp_1 = np.array([])
		for i in range( int( len(train_feature) ) ):
			tmp_2 = np.array([])
			tmp_2 = np.append( tmp_2, test[ num*18 + train_feature[i][0] ] )

			tmp_1 = np.append( tmp_1, tmp_2 )

		feature = np.append( feature, tmp_1 )

	feature = feature.reshape( ( 240, num_feature*9 ) )

	ans = np.dot(feature, w) + b

	f = open( sys.argv[3], 'w' )
	print( 'id,value', file = f )
	for i in range(240):
		print('id_', i, ',', float( ans[i] ), sep = '', file = f)
	f.close()


( out ) = read_train_data( train_feature, num_feature )

ans = out[:, 0]
data = out[:, 1:]
	
b = 0.0
w = np.zeros(num_feature * 9)

b_lr = 0.0
w_lr = np.zeros(num_feature * 9)


for i in range(it):

	b_grad = 0.0
	w_grad = np.zeros(num_feature * 9)
	loss = 0.0

	delta = ans - b - np.dot(data, w)

	loss = np.sqrt( np.sum( delta**2 ) / len(data) )
	# print(i, loss)

	b_grad = (-2) * np.sum(delta)
	w_grad = (-2) * np.dot( delta.T, data )

	b_lr += b_grad**2;			b -= lr/np.sqrt(b_lr) * b_grad
	w_lr += w_grad**2;			w -= lr/np.sqrt(w_lr) * w_grad


predit(train_feature, num_feature, b, w)