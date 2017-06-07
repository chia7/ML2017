import numpy as np
import sys
from keras.models import load_model


def load_data():
	test = np.genfromtxt( sys.argv[1] + 'test.csv', delimiter=',' )[1:, 1:]

	return test



x_test = load_data()
model = load_model("model.h5")



ans = model.predict( [x_test[:, 1], x_test[:, 0]] )



f = open(sys.argv[2], "w")
print("TestDataID,Rating", file=f)
for i in range(1, 100337):
	print(i, ans[i-1][0], sep=",", file=f)

f.close()


