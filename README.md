# Simple Neural Network for Iris Classification

This project implements a simple feed-forward neural network using TensorFlow and Keras to classify iris species based on the famous Iris dataset. This network uses one hidden layer and is trained to classify iris flowers into three species.

## Dataset
The Iris dataset is a well-known dataset for classification tasks, containing 150 samples of iris flowers with four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Model Architecture
The neural network consists of:
- Input Layer: 4 nodes (one for each feature)
- Hidden Layer: 10 nodes with ReLU activation
- Output Layer: 3 nodes (one for each class) with softmax activation

## Files in the Repository
- `neural_network.py`: Core script containing the dataset loading, neural network model, and training/evaluation functions.
- `example_notebook.ipynb`: Jupyter notebook demonstrating the model usage step-by-step, from data loading to evaluation.
- `requirements.txt`: List of dependencies to run the project.
  
## Usage

### Prerequisites
Make sure you have Python 3 installed, and install dependencies using:
```bash
pip install -r requirements.txt
