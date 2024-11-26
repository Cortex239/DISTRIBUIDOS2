import os
import numpy as np
import pandas as pd


def read_config(config_type):
    """
    Reads configuration parameters for either SAE or Softmax.
    Args:
        config_type (str): Either 'sae' or 'softmax'.
    Returns:
        dict: Configuration parameters.
    """
    try:
        config_path = os.path.join('', f'config_{config_type}.csv')
        config = np.loadtxt(config_path, delimiter=',')
        if config_type == 'sae':
            return {
                'hidden_nodes1': int(config[0]),
                'hidden_nodes2': int(config[1]),
                'penalty': float(config[2]),
                'num_runs': int(config[3]),
            }
        elif config_type == 'softmax':
            return {
                'max_iter': int(config[0]),
                'batch_size': int(config[1]),
                'learning_rate': float(config[2]),
            }
        else:
            raise ValueError(f"Invalid config_type: {config_type}")
    except Exception as e:
        raise Exception(f"Error reading config_{config_type}.csv: {str(e)}")


def load_and_preprocess_data(filename, idx_file='idx_igain.csv'):
    """
    Loads and preprocesses data using information gain indices.
    Args:
        filename (str): Data file path.
        idx_file (str): Information gain index file path.
    Returns:
        tuple: Processed features (X) and binary labels (y).
    """
    try:
        # Load data
        data = pd.read_csv(filename, header=None)
        indices = pd.read_csv(idx_file, header=None).values.flatten() - 1

        # Split into features and labels
        X = data.iloc[:, :-1].values  # All columns except the last
        y = data.iloc[:, -1].values  # Last column

        # Select relevant features
        X_selected = X[:, indices]

        # Convert labels to binary format
        y_binary = label_binary(y)

        return X_selected, y_binary
    except Exception as e:
        raise Exception(f"Error loading data {filename}: {str(e)}")


def label_binary(y):
    """
    Convert numeric labels to binary format.
    Args:
        y (numpy array): Numeric labels.
    Returns:
        numpy array: Binary encoded labels.
    """
    y_binary = np.zeros((len(y), 2))
    y_binary[y == 1, 0] = 1  # Class 1: [1, 0]
    y_binary[y == 2, 1] = 1  # Class 2: [0, 1]
    return y_binary


def sigmoid(x):
    """
    Sigmoid activation function.
    Args:
        x (numpy array): Input array.
    Returns:
        numpy array: Sigmoid output.
    """
    return 1 / (1 + np.exp(-x))


def calculate_pseudo_inverse(H, C):
    """
    Calculate pseudo-inverse using singular value decomposition (SVD).
    Args:
        H (numpy array): Hidden layer matrix.
        C (float): Regularization parameter.
    Returns:
        numpy array: Pseudo-inverse matrix.
    """
    try:
        HHT = H @ H.T  # H * H^T
        I = np.eye(HHT.shape[0])
        A = HHT + I / C
        A_inv = np.linalg.inv(A)  # Regularized inversion
        return H.T @ A_inv
    except Exception as e:
        raise Exception(f"Error in pseudo-inverse calculation: {str(e)}")


def mtx_confusion(y_true, y_pred):
    """
    Calculate confusion matrix and F-scores.
    Args:
        y_true (numpy array): True binary labels.
        y_pred (numpy array): Predicted probabilities or labels.
    Returns:
        tuple: Confusion matrix and F-scores.
    """
    y_true_class = np.argmax(y_true, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true_class, y_pred_class):
        cm[t, p] += 1

    f_scores = []
    for i in range(2):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f_scores.append(f_score)

    return cm, np.array(f_scores)


def save_outputs(costs, weights, prefix=''):
    """
    Save training outputs to files.
    Args:
        costs (list or numpy array): List of cost values.
        weights (list of numpy arrays): List of weight matrices.
        prefix (str): Prefix for file naming.
    """
    try:
        # Save costs
        cost_path = f'{prefix}costo.csv'
        np.savetxt(cost_path, costs, delimiter=',', fmt='%.6f')

        # Save weights
        for i, w in enumerate(weights, 1):
            weight_path = f'{prefix}w{i}.npy'
            np.save(weight_path, w)
    except Exception as e:
        raise Exception(f"Error saving outputs: {str(e)}")


def load_weights():
    """
    Load trained weights from the current directory.
    Returns:
        list of numpy arrays: List of weight matrices.
    """
    try:
        weights = []
        for i in range(1, 4):  # Load w1, w2, w3
            weight_path = f'w{i}.npy'
            weights.append(np.load(weight_path))
        return weights
    except Exception as e:
        raise Exception(f"Error loading weights: {str(e)}")


def save_test_outputs(confusion_matrix, f_scores):
    """
    Save test outputs to files.
    Args:
        confusion_matrix (numpy array): Confusion matrix.
        f_scores (numpy array): F-scores.
    """
    try:
        # Save confusion matrix
        conf_path = 'confusion.csv'
        np.savetxt(conf_path, confusion_matrix, delimiter=',', fmt='%d')

        # Save F-scores
        fscore_path = 'fscores.csv'
        np.savetxt(fscore_path, f_scores, delimiter=',', fmt='%.4f')
    except Exception as e:
        raise Exception(f"Error saving test outputs: {str(e)}")