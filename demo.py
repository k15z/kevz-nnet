import nnet
import nnet.layers
import numpy as np

x = np.array([
	[ 1.0,  1.0],
	[ 1.0, -1.0],
	[-1.0,  1.0],
	[-1.0, -1.0]
])
y = np.array([
	[ 1.0, 0.0],
	[ 0.0, 1.0],
	[ 0.0, 1.0],
	[ 1.0, 0.0]
])

model = nnet.Model([
	nnet.layers.ReLU(2, 10),
	nnet.layers.Softmax(10, 2)
], loss='xentropy')
model.fit(x, y, epochs=100)
print(model.predict(x))
