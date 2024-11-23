# Testing : EDL
import numpy as np
import pandas as pd
import utility as ut  # Suponiendo que utility contiene funciones relevantes

def forward_edl():
    """
    Simula el proceso de forward utilizando datos de entrada.
    Aquí se deberían incluir las operaciones de evaluación según el modelo EDL.
    """
    # Ejemplo de datos simulados: Matriz de confusión (valores aleatorios de prueba)
    confusion_matrix = np.array([[50, 10],  # TP, FP
                                  [5, 35]]) # FN, TN
    # Calcular métricas a partir de la matriz de confusión
    tp, fp, fn, tn = confusion_matrix.ravel()
    
    # Precision (P)
    precision = tp / (tp + fp)
    
    # Recall (R)
    recall = tp / (tp + fn)
    
    # F-score (F)
    fscore = 2 * (precision * recall) / (precision + recall)
    
    # Accuracy (Acc)
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    # Crear salidas
    results = {
        "confusion_matrix": confusion_matrix,
        "fscore": np.array([fscore, accuracy])  # Ejemplo: vector [F, Acc]
    }
    
    return results

def main():
    # Ejecutar forward_edl y guardar resultados
    results = forward_edl()
    
    # Guardar matriz de confusión en confusion.csv
    confusion_df = pd.DataFrame(results["confusion_matrix"],
                                columns=["Predicción P", "Predicción N"],
                                index=["Valor P", "Valor N"])
    confusion_df.to_csv("confusion.csv", index=True)

    # Guardar fscore en fscores.csv
    fscore_df = pd.DataFrame([results["fscore"]], columns=["F-score", "Accuracy"])
    fscore_df.to_csv("fscores.csv", index=False)
    
    print("Archivos confusion.csv y fscores.csv generados exitosamente.")

if __name__ == '__main__':
    main()
