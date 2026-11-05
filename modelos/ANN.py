import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layer_size=10, learning_rate=0.10, random_state=None):
        self.h = hidden_layer_size
        self.ln = learning_rate
        self.random_state = random_state

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)

        self.W1 = rng.normal(0, np.sqrt(2 / self.n_feat), (self.h, self.n_feat))
        self.b1 = np.zeros((self.h, 1))

        self.W2 = rng.normal(0, np.sqrt(2 / self.h), (self.n_classes, self.h))
        self.b2 = np.zeros((self.n_classes, 1))

    def _feedforward(self, W1, b1, W2, b2):
        def relu(x):
            return np.maximum(0, x)

        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)

        Z1 = W1.dot(self.X) + b1
        A1 = relu(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)

        return Z1, A1, Z2, A2

    def _backpropagation(self, Z1, A1, A2):
        def relu_deriv(x):
            return x > 0

        dZ2 = A2 - self.y
        dW2 = 1/self.m * dZ2.dot(A1.T)
        db2 = (1/self.m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * relu_deriv(Z1)
        dW1 = 1/self.m * dZ1.dot(self.X.T)
        db1 = (1/self.m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def _update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.ln * dW1
        self.b1 -= self.ln * db1
        self.W2 -= self.ln * dW2
        self.b2 -= self.ln * db2

    @staticmethod
    def _cross_entropy(A2, Y, eps=1e-12):
        m = Y.shape[1]
        return -np.sum(Y * np.log(np.clip(A2, eps, 1.0))) / m

    def fit(self, X_train, y_train, epochs=1000, verbose=False):
        self.m, self.n_feat = X_train.shape
        self.n_classes = len(np.unique(y_train))

        self.X = X_train.T
        self.y = np.eye(self.n_classes, dtype=int)[y_train].T

        self._init_params()

        self.history = {"loss": [], "accuracy": [], "cost": [], "acc": []}

        for i in range(epochs):
            Z1, A1, _, A2 = self._feedforward(self.W1, self.b1, self.W2, self.b2)

            loss = self._cross_entropy(A2, self.y)
            accuracy = np.mean(np.argmax(A2, axis=0) == np.argmax(self.y, axis=0))
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)
            self.history["cost"].append(loss)
            self.history["acc"].append(accuracy)

            dW1, db1, dW2, db2 = self._backpropagation(Z1, A1, A2)
            self._update_params(dW1, db1, dW2, db2)

            if verbose and ((i + 1) % 50 == 0 or i == 0):
                print(f"Época {i + 1}/{epochs} - Acurácia: {accuracy:.2%}")

        self.n_epochs_ = len(self.history["loss"])
        return self

    def predict_proba(self, X):
        Z1 = self.W1.dot(X.T) + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = self.W2.dot(A1) + self.b2
        exp_z = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)