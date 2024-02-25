input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  np.random.seed(0)
  return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
  np.random.seed(1)
  X = 10 * np.random.randn(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

net = init_toy_model()
X, y = init_toy_data()

# First layer pre-activation
z1 = X.dot(W1) + b1

# First layer activation
a1 = np.maximum(0, z1)

# Second layer pre-activation
z2 = a1.dot(W2) + b2

scores = z2
#The scores variable keeps the pre-activation values for the output layer. We will be using this in a while to find the activation values in the output layer, and consequently the cross-entropy loss.

#So for the second-layer activation, we have:
# Second layer activation
exp_scores = np.exp(scores)
a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

#And to compute for the loss, we perform the following code:
corect_logprobs = -np.log(a2[range(N), y])
data_loss = np.sum(corect_logprobs) / N
reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
loss = data_loss + reg_loss

#We now implement the backward pass, where we compute the derivatives of the weights and biases and propagate them across the network. In this way, the network gets a feel of the contributions of each individual units, and adjusts itself accordingly so that the weights and biases are optimal.
#We first compute for the gradients, thus we have:
dscores = a2
dscores[range(N),y] -= 1
dscores /= N

#And then we propagate them back to our network:
# W2 and b2
grads['W2'] = np.dot(a1.T, dscores)
grads['b2'] = np.sum(dscores, axis=0)

# Propagate to hidden layer
dhidden = np.dot(dscores, W2.T)

# Backprop the ReLU non-linearity
dhidden[a1 <= 0] = 0

# Finally into W,b
grads['W1'] = np.dot(X.T, dhidden)
grads['b1'] = np.sum(dhidden, axis=0)

#We should not also forget to regularize our gradients:
grads['W2'] += reg * W2
grads['W1'] += reg * W1
#In our TwoLayerNet class, we also implement a train() function that trains the neural network using stochastic gradient descent. First, we create a random minibatch of training data and labels, then we store them in X_batch and Y_batch respectively:
sample_indices = np.random.choice(np.arange(num_train), batch_size)
X_batch = X[sample_indices]
y_batch = y[sample_indices]
#And then we update our parameters in our network.
self.params['W1'] += -learning_rate * grads['W1']
self.params['b1'] += -learning_rate * grads['b1']
self.params['W2'] += -learning_rate * grads['W2']
self.params['b2'] += -learning_rate * grads['b2']

#Lastly, we implement a predict() function that classifies our inputs with respect to the scores and activations found after the output layer. We simply make a forward pass for the input, and then get the maximum of the scores that was found.
z1 = X.dot(self.params['W1']) + self.params['b1']
a1 = np.maximum(0, z1) # pass through ReLU activation function
scores = a1.dot(self.params['W2']) + self.params['b2']
y_pred = np.argmax(scores, axis=1)

#Once we’ve implemented our functions, we can then test them to see if they are working properly. The IPython notebook tests our implementation by checking our computed scores with respect to the correct scores hardcoded in the program.
#In the first test, we got a difference of 3.68027206479e-08
#output
#Your scores:
#[[-0.81233741 -1.27654624 -0.70335995]
 #[-0.17129677 -1.18803311 -0.47310444]
# [-0.51590475 -1.01354314 -0.8504215 ]
 #[-0.15419291 -0.48629638 -0.52901952]
 #[-0.00618733 -0.12435261 -0.15226949]]

#correct scores:
#[[-0.81233741 -1.27654624 -0.70335995]
 #[-0.17129677 -1.18803311 -0.47310444]
 #[-0.51590475 -1.01354314 -0.8504215 ]
 #[-0.15419291 -0.48629638 -0.52901952]
 #[-0.00618733 -0.12435261 -0.15226949]]