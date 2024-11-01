import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

class IrisDataset:
    def __init__(self):
        # Load and preprocess the dataset
        self.data, self.targets = load_iris(return_X_y=True)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
    
    def get_data(self, test_size=0.2):
        """Splits data into training and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.targets, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

class SimpleNeuralNetwork(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,))
        self.fc2 = tf.keras.layers.Dense(output_size, activation='softmax')
    
    def call(self, inputs):
        x = self.fc1(inputs)
        return self.fc2(x)

def train(model, X_train, y_train, epochs=100, batch_size=16):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history

def evaluate(model, X_test, y_test):
    predictions = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Hyperparameters
    input_size = 4       # Number of features in Iris dataset
    hidden_size = 10     # Size of hidden layer
    output_size = 3      # Number of classes (3 types of iris flowers)
    epochs = 100

    # Load data
    dataset = IrisDataset()
    X_train, X_test, y_train, y_test = dataset.get_data()

    # Initialize model
    model = SimpleNeuralNetwork(input_size, hidden_size, output_size)

    # Train and evaluate the model
    train(model, X_train, y_train, epochs)
    evaluate(model, X_test, y_test)
