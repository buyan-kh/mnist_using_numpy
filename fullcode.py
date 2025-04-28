import numpy as np
data = np.load('mnist.npz')
X_train = data['x_train']
y_train = data['y_train']
X_test = data['x_test']
y_test = data['y_test']

def one_hot_encoding(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

y_train_encoded = one_hot_encoding(y_train)
y_test_encoded = one_hot_encoding(y_test)

X_train = X_train.T
X_test = X_test.T

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_pass(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def compute_loss(self, A2, Y):
        m = Y.shape[1]
        log_probs = np.log(A2 + 1e-8)
        loss = -np.sum(Y * log_probs) / m
        return loss

    def backward_pass(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, Y, epochs=20):
        for epoch in range(epochs):
            Z1, A1, Z2, A2 = self.forward_pass(X)
            loss = self.compute_loss(A2, Y)
            dW1, db1, dW2, db2 = self.backward_pass(X, Y, Z1, A1, Z2, A2)
            self.update_parameters(dW1, db1, dW2, db2)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        _, _, _, A2 = self.forward_pass(X)
        predictions = np.argmax(A2, axis=0)
        return predictions

    def evaluate(self, X, Y_true):
        preds = self.predict(X)
        accuracy = np.mean(preds == Y_true) * 100
        return accuracy

    def save_model(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    
    def load_model(self, filename):
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10, learning_rate=0.1)

model.train(X_train, y_train_encoded.T, epochs=300)

train_accuracy = model.evaluate(X_train, y_train)
test_accuracy = model.evaluate(X_test, y_test)

model.save_model('mnist_trained_model.npz')

# model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
# model.load_model('mnist_trained_model.npz')

print(f"Train Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")