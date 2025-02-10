import pickle
import numpy as np
import nltk
from utils import extract_features

nltk.download('twitter_samples')
nltk.download('stopwords')

# Q1.1 Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Q1.2 One Layer Inference
def inference_layer(X, W, b):
    return sigmoid(np.dot(W, X) + b)

# Q1.3 Two Layer Inference
def inference_2layers(X, W1, W2, b1, b2):
    h = sigmoid(np.dot(W1, X) + b1)
    return sigmoid(np.dot(W2, h) + b2)

# Q1.4 Binary Cross-Entropy Loss
def bce_forward(yhat, y):
    epsilon = 1e-8
    yhat = np.clip(yhat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

# Q3.1 Gradients Calculation
def gradients(X, y, W1, W2, b1, b2):
    h = sigmoid(np.dot(W1, X) + b1)
    yhat = sigmoid(np.dot(W2, h) + b2)
    loss = bce_forward(yhat, y)

    dL_dyhat = yhat - y
    dL_dW2 = np.dot(dL_dyhat, h.T) / X.shape[1]
    dL_db2 = np.mean(dL_dyhat, axis=1, keepdims=True)
    dL_dh = np.dot(W2.T, dL_dyhat) * h * (1 - h)
    dL_dW1 = np.dot(dL_dh, X.T) / X.shape[1]
    dL_db1 = np.mean(dL_dh, axis=1, keepdims=True)

    return dL_dW1, dL_dW2, dL_db1, dL_db2, loss

# Q4 Parameter Updates
def update_params(W1, b1, W2, b2, dW1, dW2, db1, db2, lr=0.01):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

# Q5 Training the Neural Network
def train_nn(filename, hidden_layer_size=10, iters=1000, lr=0.01):
    """
    Trains the neural network using gradient descent with feature extraction.

    Args:
      filename (str): Path to dataset.
      hidden_layer_size (int): Number of hidden units.
      iters (int): Number of iterations.
      lr (float): Learning rate.

    Returns:
      Trained parameters (W1, b1, W2, b2).
    """
    # Load dataset
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    train_x_raw = data['train_x']  # Raw tweets (text)
    train_y = data['train_y']      # Labels (0 or 1)
    freqs = data['freqs']          # Word frequency dictionary

    # Extract features from tweets
    train_x = np.array([extract_features(tweet, freqs) for tweet in train_x_raw]).T  # Transpose for correct shape

    # Initialize parameters
    d = train_x.shape[0]  # Number of features
    H = hidden_layer_size  # Hidden layer size

    W1 = np.random.randn(H, d) * 0.01
    b1 = np.zeros((H, 1))
    W2 = np.random.randn(1, H) * 0.01
    b2 = np.zeros((1, 1))

    # Training loop
    for i in range(iters):
        dW1, dW2, db1, db2, loss = gradients(train_x, train_y, W1, W2, b1, b2)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, dW2, db1, db2, lr)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return W1, b1, W2, b2


# Save Model Parameters
def save_model(W1, b1, W2, b2, filename='assignment2.pkl'):
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
