import numpy as np
import utility as ut  # Funciones auxiliares como sigmoid y softmax


# Preprocesar datos
def preprocess_data(file_dtrain, file_igain, output_file):
    """
    Preprocesa los datos iniciales para convertir las clases y seleccionar características relevantes.
    Guarda el archivo preprocesado como DataTrain.csv.
    """
    data = np.loadtxt(file_dtrain, delimiter=',', skiprows=1, dtype=float)
    X = data[:, :-1]  # Características
    y = data[:, -1].astype(int)  # Etiquetas (como enteros)
    
    # Convertir clases a binarias: Normal (1,0) y Ataques (0,1)
    y_binary = np.zeros((y.shape[0], 2))
    y_binary[np.where(y == 1), 0] = 1.0  # Clase normal
    y_binary[np.where(y == 2), 1] = 1.0  # Clase ataque
    
    # Seleccionar características relevantes usando idx_igain.csv
    selected_features = np.loadtxt(file_igain, delimiter=',', dtype=int)
    selected_features = selected_features[selected_features < X.shape[1]]  # Filtrar índices inválidos
    X = X[:, selected_features]
    
    # Guardar DataTrain.csv
    processed_data = np.hstack((X, y_binary))
    np.savetxt(output_file, processed_data, delimiter=',', header=",".join([f"var{i}" for i in range(X.shape[1])]) + ",class1,class2", comments='')
    
    return X, y_binary


# Entrenar SAE-ELM
def train_sae_elm(X, y, config_file):
    """
    Entrena SAE-ELM con múltiples ejecuciones y selecciona la mejor según el error cuadrático medio.
    """
    config = np.loadtxt(config_file, delimiter=',', dtype=float)
    n_hidden1 = int(config[0])  # Capa oculta 1
    n_hidden2 = int(config[1])  # Capa oculta 2
    penalty_factor = config[2]  # Factor de penalización
    n_runs = int(config[3])  # Número de ejecuciones
    
    best_beta, best_H2 = None, None
    min_error = float('inf')
    
    for run in range(n_runs):
        # Inicializar pesos y calcular activaciones
        W1 = np.random.uniform(-1, 1, (X.shape[1], n_hidden1))
        H1 = ut.sigmoid(np.dot(X, W1))
        W2 = np.random.uniform(-1, 1, (n_hidden1, n_hidden2))
        H2 = ut.sigmoid(np.dot(H1, W2))
        
        # Calcular Beta con pseudo-inversa regularizada
        regularization = penalty_factor * np.eye(H2.shape[1])
        beta = np.dot(np.linalg.pinv(np.dot(H2.T, H2) + regularization), np.dot(H2.T, y))
        
        # Calcular error cuadrático medio
        y_pred = np.dot(H2, beta)
        error = np.mean((y - y_pred) ** 2)
        
        if error < min_error:
            min_error = error
            best_beta = beta
            best_H2 = H2
        
        print(f"Ejecución {run + 1}/{n_runs}: Error = {error}")
    
    return best_H2, best_beta


# Entrenar capa Softmax
def train_softmax(H, y, config_file):
    """
    Entrena la capa Softmax usando mADAM y guarda los costos y pesos.
    """
    config = np.loadtxt(config_file, delimiter=',', dtype=float)
    max_iterations = int(config[0])
    batch_size = int(config[1])
    learning_rate = config[2]
    
    L, m = y.shape[1], H.shape[1]
    weights = np.random.uniform(-1, 1, (L, m))  # Inicializar pesos
    
    # Parámetros mADAM
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    mt, vt = np.zeros_like(weights), np.zeros_like(weights)
    costs = []
    
    for iteration in range(1, max_iterations + 1):
        # Mini-batch
        for i in range(0, H.shape[0], batch_size):
            X_batch = H[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            # Forward pass
            logits = np.dot(X_batch, weights.T)
            probabilities = ut.softmax(logits)
            
            # Backward pass
            gradient = np.dot((probabilities - y_batch).T, X_batch) / batch_size
            mt = beta1 * mt + (1 - beta1) * gradient
            vt = beta2 * vt + (1 - beta2) * (gradient ** 2)
            mt_hat = mt / (1 - beta1 ** iteration)
            vt_hat = vt / (1 - beta2 ** iteration)
            weights -= learning_rate * mt_hat / (np.sqrt(vt_hat) + epsilon)
        
        # Calcular costo
        logits = np.dot(H, weights.T)
        probabilities = ut.softmax(logits)
        cost = -np.mean(np.sum(y * np.log(probabilities + epsilon), axis=1))
        costs.append(cost)
        if iteration % 100 == 0:
            print(f"Iteración {iteration}/{max_iterations}: Costo = {cost}")
    
    # Guardar costos y pesos
    np.savetxt("costo.csv", costs, delimiter=',')
    np.savetxt("pesos.csv", weights, delimiter=',')
    
    return weights


# Programa principal
def main():
    file_dtrain = "dtrain.csv"
    file_igain = "idx_igain.csv"
    output_file = "DataTrain.csv"
    config_sae = "config_sae.csv"
    config_softmax = "config_softmax.csv"
    
    # Preprocesar datos
    X, y = preprocess_data(file_dtrain, file_igain, output_file)
    
    # Entrenar SAE-ELM
    H, beta = train_sae_elm(X, y, config_sae)
    
    # Entrenar Softmax
    softmax_weights = train_softmax(H, y, config_softmax)
    
    print("Entrenamiento completado. Archivos generados: DataTrain.csv, costo.csv, pesos.csv")


if __name__ == "__main__":
    main()
