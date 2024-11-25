import numpy as np
import pandas as pd
import utility as ut

def load_data(data_path, idx_igain_path):
    try:
        data = pd.read_csv(data_path)
        idx_igain = pd.read_csv(idx_igain_path, header=None).values.flatten()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error cargando archivos: {e}")
    
    # Seleccionar columnas basadas en idx_igain
    data = data.iloc[:, idx_igain]

    # Transformar clases numéricas a binarias y asignar una nueva columna
    data['binary_labels'] = data.iloc[:, -1].astype(int).apply(lambda x: [1, 0] if x == 1 else [0, 1])
    data.drop(data.columns[-2], axis=1, inplace=True)  # Eliminar la columna original
    data.rename(columns={'binary_labels': data.columns[-1]}, inplace=True)  # Renombrar la nueva columna

    return data

def train_sae_elm(data, config):
    # Cargar configuración SAE-ELM
    hidden_layer_1, hidden_layer_2, penalty, num_runs = pd.read_csv(config, header=None).values.flatten()

    np.random.seed(42)
    weights_1 = np.random.randn(data.shape[1] - 1, hidden_layer_1)  # (n_features, hidden_layer_1)
    weights_2 = np.random.randn(hidden_layer_1, hidden_layer_2)  # (hidden_layer_1, hidden_layer_2)

    # Proyección en la primera capa oculta
    H = ut.activation_function(data.iloc[:, :-1].values @ weights_1, type="sigmoid")  # (3999, 20)

    # Calcular la pseudo-inversa correctamente
    pseudo_inverse = ut.pseudo_inverse(H, penalty)  # Forma (3999, 20)

    print(f"pseudo_inverse shape: {pseudo_inverse.shape}")
    print(f"weights_2 shape: {weights_2.shape}")

    # Corregir el cálculo de pesos óptimos
    labels = np.array(data.iloc[:, -1].tolist())
    weights_optimal = pseudo_inverse @ labels  # (20, 3999) * (3999, 2) = (20, 2)

    if weights_optimal.shape != (hidden_layer_1, labels.shape[1]):
        raise ValueError(f"weights_optimal generado con dimensiones incorrectas: {weights_optimal.shape}")
    
    return weights_1, weights_optimal

def train_softmax(data, config_softmax_path, weights_1, weights_2):
    config = pd.read_csv(config_softmax_path, header=None).values.flatten()
    max_iter, batch_size = map(int, config[:2])
    learning_rate = float(config[2])
    
    hidden_layer_2 = 2  # Debería ser el tamaño de la capa oculta 2
    n_classes = 2  # Número de clases en la salida
    weights = np.random.randn(hidden_layer_2, n_classes)  # (2, 2)
    
    m, v = 0, 0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    log_epsilon = 1e-10  # Small value to avoid log(0)
    
    costs = []
    for i in range(max_iter):
        indices = np.random.choice(data.index, batch_size, replace=False)
        X_batch = data.iloc[indices, :-1].values
        y_batch = np.array(data.iloc[indices, -1].tolist())
        
        # Pasar a través de la primera capa oculta
        H1 = ut.activation_function(X_batch @ weights_1, type="sigmoid")
        
        # Pasar a través de la segunda capa oculta
        H2 = ut.activation_function(H1 @ weights_2, type="sigmoid")
        
        logits = H2 @ weights
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        cost = -np.mean(np.sum(np.log(probabilities + log_epsilon) * y_batch, axis=1))
        costs.append(cost)
        
        gradients = H2.T @ (probabilities - y_batch) / batch_size
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * gradients ** 2
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
    return weights, costs


def main():
    file_dtrain = "dtrain.csv"
    file_igain = "idx_igain.csv"
    config_sae = "config_sae.csv"
    config_softmax = "config_softmax.csv"

    # Cargar datos
    data = load_data(file_dtrain, file_igain)
    
    # Entrenar SAE-ELM
    weights_1, best_w2 = train_sae_elm(data, config_sae)

    # Entrenar Softmax
    weights_softmax, costs = train_softmax(data, config_softmax, weights_1, best_w2)

    # Guardar salidas
    pd.DataFrame(costs).to_csv('costo.csv', index=False, header=False)
    pd.DataFrame(weights_1).to_csv('w1.csv', index=False, header=False)
    pd.DataFrame(best_w2).to_csv('w2.csv', index=False, header=False)
    pd.DataFrame(weights_softmax).to_csv('w3.csv', index=False, header=False)

    print("Entrenamiento completado. Archivos generados: costo.csv, w1.csv, w2.csv, w3.csv")

if __name__ == "__main__":
    main()

