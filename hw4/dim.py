import numpy as np
from keras.models import load_model
import math
import sys

data = np.load( sys.argv[1] )
x_test = np.array([])
for i in range(200):
	x = data[ str(i) ].T
	val, vec = np.linalg.eigh( np.cov(x) )
	x_test = np.append(x_test, val)

x_test = x_test.reshape((200, 100))[:, 40:]


model = load_model( "model.h5" )
ans = model.predict(x_test)

f = open( sys.argv[2], "w")
print("SetId,LogDim", file = f)
for i in range(200):
	print(i, np.log(round(ans[i,0])), sep = ',', file = f)
f.close()
