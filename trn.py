import numpy as np
import pandas as pd
import utility as ut
import time
import os  # Importar el módulo os

def load_data(data_path, idx_igain_path):
    """Carga y transforma los datos de entrenamiento."""
    try:
        data = pd.read_csv(data_path)
        idx_igain = pd.read_csv(idx_igain_path, header=None).values.flatten()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error cargando archivos: {e}")
    
    if max(idx_igain) >= data.shape[1]:
        raise ValueError("Los índices de idx_igain.csv exceden las columnas en dtrain.csv.")
    
    # Seleccionar columnas relevantes
    data = data.iloc[:, idx_igain]

    # Transformar clases numéricas a binarias y asignar una nueva columna
    data['binary_labels'] = data.iloc[:, -1].apply(lambda x: [1, 0] if x == 1 else [0, 1])
    data.drop(columns=data.columns[-2], inplace=True)  # Eliminar la columna original
    data.rename(columns={'binary_labels': data.columns[-1]}, inplace=True)  # Renombrar la nueva columna

    return data

def train_sae_layer(data, hidden_nodes, penalty, num_runs):
    """Entrena una capa SAE utilizando ELM con pseudo-inversa"""
    try:
        n_samples, n_features = data.shape
        best_error = float('inf')
        best_weights = None

        r = np.sqrt(6 / (hidden_nodes + n_features))

        for run in range(num_runs):
            w = np.random.uniform(-r, r, (n_features, hidden_nodes))
            H = ut.sigmoid(data @ w)
            H_pinv = ut.calculate_pseudo_inverse(H, penalty)
            w_out = H_pinv @ data
            reconstructed = H @ w_out
            error = np.mean((data - reconstructed) ** 2)

            if error < best_error:
                best_error = error
                best_weights = w

            print(f"Run {run + 1}/{num_runs}, Error: {error:.6f}")

        print(f"Best reconstruction error: {best_error:.6f}")
        return best_weights

    except Exception as e:
        raise Exception(f"Error in SAE layer training: {str(e)}")

def train_softmax(data, labels, config):
    """Entrena la capa Softmax utilizando mini-batch mAdam con estabilidad numérica"""
    n_features = data.shape[1]
    n_classes = labels.shape[1]
    w = np.random.randn(n_features, n_classes) * 0.01
    m, v = np.zeros_like(w), np.zeros_like(w)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    costs = []

    patience, min_delta, min_epochs = 10, 1e-5, 200
    best_cost, patience_counter = float('inf'), 0
    best_weights = None
    n_batches = data.shape[0] // config['batch_size']

    for epoch in range(config['max_iter']):
        idx = np.random.permutation(data.shape[0])
        data_shuffled, labels_shuffled = data[idx], labels[idx]
        epoch_cost = 0

        for i in range(n_batches):
            start_idx, end_idx = i * config['batch_size'], (i + 1) * config['batch_size']
            X_batch, y_batch = data_shuffled[start_idx:end_idx], labels_shuffled[start_idx:end_idx]
            logits = X_batch @ w
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            batch_cost = -np.mean(np.sum(y_batch * np.log(probs + epsilon), axis=1))
            epoch_cost += batch_cost
            grad = (1 / config['batch_size']) * X_batch.T @ (probs - y_batch)
            m, v = beta1 * m + (1 - beta1) * grad, beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat, v_hat = m / (1 - beta1 ** (epoch + 1)), v / (1 - beta2 ** (epoch + 1))
            w -= config['learning_rate'] * m_hat / (np.sqrt(v_hat) + epsilon)

        epoch_cost /= n_batches
        costs.append(epoch_cost)

        if epoch_cost < best_cost:
            best_cost, best_weights = epoch_cost, w.copy()

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{config['max_iter']}, Cost: {epoch_cost:.6f}")

        if epoch >= min_epochs:
            if epoch_cost < best_cost - min_delta:
                best_cost, patience_counter = epoch_cost, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, Best cost: {best_cost:.6f}")
                    break

    return best_weights, np.array(costs)

def train_edl():
    """Entrena el modelo EDL completo"""
    start_time = time.time()

    try:
        sae_config = ut.read_config_sae()
        softmax_config = ut.read_config_softmax()
        print("Loading and preprocessing data...")
        X_train, y_train = ut.load_and_preprocess_data('dtrain.csv')
        print("Normalizing data...")
        mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0) + 1e-10
        X_train_norm = (X_train - mean) / std

        # Crear el directorio 'output' si no existe
        if not os.path.exists('output'):
            os.makedirs('output')

        np.save('output/mean.npy', mean)
        np.save('output/std.npy', std)
        
        # Entrenamiento de la primera capa SAE
        print("Training first SAE layer...")
        w1 = train_sae_layer(X_train_norm, sae_config['hidden_nodes1'], sae_config['penalty'], sae_config['num_runs'])
        H1 = ut.sigmoid(X_train_norm @ w1)
        
        # Entrenamiento de la segunda capa SAE
        print("Training second SAE layer...")
        w2 = train_sae_layer(H1, sae_config['hidden_nodes2'], sae_config['penalty'], sae_config['num_runs'])
        H2 = ut.sigmoid(H1 @ w2)
        
        # Entrenamiento de la capa Softmax
        print("Training Softmax layer...")
        w3, costs = train_softmax(H2, y_train, softmax_config)
        
        # Guardar pesos y costos
        weights = [w1, w2, w3]
        ut.save_outputs(costs, weights)
        
        # Verificación del tiempo de ejecución
        execution_time = time.time() - start_time
        if execution_time > 120:  # 2 minutos
            print(f"Warning: Execution time ({execution_time:.2f}s) exceeded 2 minutes limit")
        
        return weights, costs

    except Exception as e:
        print(f"Error in train_edl: {str(e)}")
        return None, None

def main():
    print("Starting EDL training...")
    weights, costs = train_edl()
    if weights is not None:
        print("Training completed successfully")
    else:
        print("Training failed")

if __name__ == '__main__':
    main()
