from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

arr = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

face = np.array([])
for ch in arr:
	path = 'faceExpressionDatabase/' + ch + '0'
	for i in range(10):
		im = Image.open( path + str(i) + ".bmp")
		face = np.append(face, np.array(im.convert('L')).flatten().astype("float32") )

face = face.reshape( (10*10, 64*64) )
mean_face = face.mean(axis = 0)

S = (face - mean_face).T
U, s, V = np.linalg.svd(S)



# problem 1.1
nb_eigenface = 9
fig = plt.figure(figsize=(5, 5))
for i in range(nb_eigenface):
	ax = fig.add_subplot(nb_eigenface/3, 3, i+1)
	ax.imshow(U[:, i].reshape(64, 64), cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()
fig.savefig("eigenfaces9")

fig = plt.figure(figsize=(8, 8))
plt.imshow(mean_face.reshape(64, 64), cmap='gray')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
fig.savefig("mean_face")



# problem 1.2
nb_eigenface = 5
fig = plt.figure(figsize=(14, 14))
for i in range(100):

	tmp = mean_face.reshape( (64, 64) )
	for k in range(nb_eigenface):
		tmp += np.dot( U[:, k].T, face[i] - mean_face ) * U[:, k].reshape( (64, 64) )
	ax = fig.add_subplot(10, 10, i+1)
	ax.imshow(tmp, cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()

fig.savefig("reconstruction5_face")


fig = plt.figure(figsize=(14, 14))
for i in range(100):
	ax = fig.add_subplot(10, 10, i+1)
	ax.imshow(face[i].reshape( (64, 64) ), cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()

fig.savefig("original_face")



# problem 1.3
tmp = np.array([mean_face]*100)
ans = 0
for nb_eigenfaces in range(64*64):

	rmse = 0
	for i in range(100):

		tmp[i] += np.dot( U[:, nb_eigenfaces].T, face[i] - mean_face ) * U[:, nb_eigenfaces]
		rmse += ( (face[i] - tmp[i])**2 ).mean()

	rmse = np.sqrt( rmse / 100 ) / 255
	if rmse < 0.01:
		ans = nb_eigenfaces + 1
		print(nb_eigenfaces + 1)
		break

fig = plt.figure(figsize=(14, 14))
for i in range(100):
	ax = fig.add_subplot(10, 10, i+1)
	ax.imshow(tmp[i].reshape( (64, 64) ), cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()
fig.savefig("reconstruction60_face")


