import numpy as np

def mse(expected, output):
	assert expected.shape == output.shape
	gradient = expected - output
	loss = np.sum(np.power(expected - output, 2))
	return (loss, gradient)

def xentropy(expected, output):
	assert expected.shape == output.shape
	gradient = expected / output
	loss = -np.sum(expected * np.log(output))
	return (loss, gradient)

class Model:
	def __init__(self, layers, loss='mse'):
		assert len(layers) > 0
		for i in range(1, len(layers)):
			assert layers[i-1].output_dims == layers[i].input_dims

		self.layers = layers
		if loss == 'mse':
			self.loss_fn = mse
		elif loss == 'xentropy':
			self.loss_fn = xentropy
		else:
			raise ValueError("Unsupported loss function.")

	def fit(self, x, y, epochs=10):
		assert x.shape[0] == y.shape[0]
		assert x.shape[1] == self.layers[0].input_dims
		assert y.shape[1] == self.layers[-1].output_dims

		losses = []
		for epoch in range(epochs):
			loss, gradient = self.loss_fn(y, self.predict(x))
			for layer in reversed(self.layers):
				gradient = layer.backpropagate(gradient)
			losses += [loss]
		return losses

	def predict(self, x):
		assert x.shape[1] == self.layers[0].input_dims

		output = x
		for layer in self.layers:
			output = layer.feedforward(output)
		return output
