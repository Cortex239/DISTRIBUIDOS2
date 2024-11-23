# My Utility: auxiliary functions

import numpy as np

def label_binary(data, class_column):
    """
    Convierte las etiquetas de clase a valores binarios.
    Asume que la columna de clase tiene valores como 'normal' o 'attack'.
    """
    binary_labels = np.where(data[class_column] == 'normal', 1.0, 0.0)
    return binary_labels

def mtx_confusion(y_true, y_pred):
    """
    Calcula la matriz de confusión.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tp, fp], [fn, tn]])

def sigmoid(x):
    """
    Calcula la función sigmoide con protección contra desbordamiento numérico.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("La entrada debe ser un arreglo de NumPy.")
    x = np.clip(x, -500, 500)  # Limita los valores extremos
    return 1 / (1 + np.exp(-x))

def softmax(z):
    """
    Calcula la función softmax con validación de entrada.
    """
    if z.ndim != 2:
        raise ValueError("La entrada a softmax debe ser una matriz 2D.")
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Evitar overflow numérico
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def mse(y_true, y_pred):
    """
    Calcula el error cuadrático medio.
    """
    return np.mean((y_true - y_pred) ** 2)

def accuracy(y_true, y_pred):
    """
    Calcula la precisión (accuracy).
    """
    correct = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    return correct / total

def pseudo_inverse(H, penalty_factor):
    """
    Calcula la pseudo-inversa de una matriz H con regularización.
    """
    regularization = max(penalty_factor, 1e-8) * np.eye(H.shape[1])  # Regularización mínima
    return np.linalg.pinv(np.dot(H.T, H) + regularization).dot(H.T)
