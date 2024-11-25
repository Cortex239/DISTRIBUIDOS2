import numpy as np
import pandas as pd
import utility as ut

def load_test_data(data_path, idx_igain_path):
    """Carga y transforma los datos de prueba."""
    try:
        data = pd.read_csv(data_path)
        idx_igain = pd.read_csv(idx_igain_path, header=None).values.flatten()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error cargando archivos: {e}")
    
    if max(idx_igain) >= data.shape[1]:
        raise ValueError("Los índices de idx_igain.csv exceden las columnas en dtest.csv.")
    
    data = data.iloc[:, idx_igain]
    labels = data.iloc[:, -1].values
    features = data.iloc[:, :-1].values
    
    # Transformar clases numéricas a binarias
    binary_labels = np.array([[1, 0] if label == 1 else [0, 1] for label in labels])
    
    return features, binary_labels

def predict(features, weights_1, weights_2, weights_softmax):
    """Genera predicciones basadas en características de entrada y pesos entrenados."""
    try:
        hidden_layer_1 = ut.activation_function(features @ weights_1, type="sigmoid")
        hidden_layer_2 = ut.activation_function(hidden_layer_1 @ weights_2, type="sigmoid")
        
        if hidden_layer_2.shape[1] != weights_softmax.shape[0]:
            raise ValueError(f"Dimensiones incompatibles: hidden_layer_2 {hidden_layer_2.shape}, weights_softmax {weights_softmax.shape}")
        
        logits = hidden_layer_2 @ weights_softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probabilities
    except Exception as e:
        raise RuntimeError(f"Error durante la predicción: {e}")

def calculate_metrics(true_labels, predictions):
    """Calcula matriz de confusión y métricas de desempeño."""
    true_labels = np.argmax(true_labels, axis=1)
    predictions = np.argmax(predictions, axis=1)
    
    tp = np.sum((true_labels == 1) & (predictions == 1))
    tn = np.sum((true_labels == 0) & (predictions == 0))
    fp = np.sum((true_labels == 0) & (predictions == 1))
    fn = np.sum((true_labels == 1) & (predictions == 0))
    cm = np.array([[tp, fp], [fn, tn]])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    
    return cm, precision, recall, f_score, accuracy

def main():
    """Función principal para cargar datos, realizar predicciones y calcular métricas."""
    try:
        features, true_labels = load_test_data('dtest.csv', 'idx_igain.csv')
        
        weights_1 = np.loadtxt('w1.csv', delimiter=',')
        weights_2 = np.loadtxt('w2.csv', delimiter=',')
        weights_softmax = np.loadtxt('w3.csv', delimiter=',')
        
        if weights_2.shape[0] != weights_1.shape[1]:
            raise ValueError(f"Incompatible shapes: weights_2 {weights_2.shape}, weights_1 {weights_1.shape}")
        
        if weights_softmax.shape[0] != weights_2.shape[1]:
            raise ValueError(f"Incompatible shapes: weights_softmax {weights_softmax.shape}, weights_2 {weights_2.shape}")
        
        predictions = predict(features, weights_1, weights_2, weights_softmax)
        
        cm, precision, recall, f_score, accuracy = calculate_metrics(true_labels, predictions)
        
        np.savetxt('confusión.csv', cm, delimiter=',', fmt='%d')
        np.savetxt('fscores.csv', np.array([f_score]), delimiter=',', fmt='%.4f')
        
        print(f"Precisión: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-score: {f_score:.4f}")
        print(f"Exactitud: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error en main: {e}")

if __name__ == "__main__":
    main()
