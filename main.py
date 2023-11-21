import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting data into training and testing set
def train_test_split(X, y, test_size=0.25, random_state=0):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_index = int((1 - test_size) * len(X))
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
def standard_scaler(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# Support Vector Machine model using PyTorch
class SVMClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.weights = torch.randn((X.shape[1], 1), dtype=torch.float32, requires_grad=True)
        self.bias = torch.randn(1, dtype=torch.float32, requires_grad=True)
        self.train()

    def train(self):
        for epoch in range(self.epochs):
            self.predictions = torch.matmul(self.X, self.weights) + self.bias
            hinge_loss = torch.sum(torch.max(0.1 - self.Y * self.predictions, torch.tensor(0.0)))
            l2_regularization = torch.sum(self.weights**2)
            loss = hinge_loss + 0.01 * l2_regularization

            loss.backward()
            with torch.no_grad():
                self.weights -= self.learning_rate * self.weights.grad
                self.bias -= self.learning_rate * self.bias.grad

                # Zero gradients
                self.weights.grad.zero_()
                self.bias.grad.zero_()

    def predict(self, X):
        return torch.sign(torch.matmul(X, self.weights) + self.bias).numpy()

svm_classifier = SVMClassifier(learning_rate=0.01, epochs=1000)
svm_classifier.fit(X_train_tensor, Y_train_tensor)

# Predicting for age=30, estimated salary=87000
new_data = torch.tensor([[30, 87000]], dtype=torch.float32)
predicted_label = svm_classifier.predict(new_data)
print(predicted_label)

# Predicting for test data
Y_pred_tensor = torch.tensor([svm_classifier.predict(x) for x in X_test_tensor], dtype=torch.float32)
Y_pred = Y_pred_tensor.numpy()
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_pred), 1)), axis=1))

# Confusion Matrix
def confusion_matrix(Y_true, Y_pred):
    cm = np.zeros((2, 2))
    for i in range(len(Y_true)):
        cm[int(Y_true[i]), int(Y_pred[i])] += 1
    return cm

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# Visualizing the Test set results
def plot_decision_boundary(X, Y, classifier, title):
    X_set, y_set = X.numpy(), Y.numpy().flatten()
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1),
                         np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1))
    plt.contourf(X1, X2, classifier.predict(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T, dtype=torch.float32)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

plot_decision_boundary(X_test_tensor, Y_test_tensor, svm_classifier, 'Support Vector Machine (Test set)')
