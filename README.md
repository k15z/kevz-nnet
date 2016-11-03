# kevz-nnet
This is a lightweight neural network library for educational use. It's much too
slow for any real-world applications but has almost no dependencies (other than
numpy) and works reasonably well.

## usage
```
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

# model.predict(x) = 
#  [[ 0.89913277  0.10086723]
#   [ 0.25143146  0.74856854]
#   [ 0.12941995  0.87058005]
#   [ 0.92600733  0.07399267]]
```
