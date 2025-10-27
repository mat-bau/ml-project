import numpy as np

def load_data(path):
    """Charge les données - Returns: DataFrame"""
    pass

def save_predictions(y_pred, path='predictions/y_pred.csv'):
    """
    Sauvegarde les prédictions au bon format
    y_pred: array de prédictions
    """
    # Format requis :
    # - Pas de header
    # - Pas de guillemets
    # - Un nombre par ligne
    np.savetxt(path, y_pred, fmt='%.6f')
    pass

def verify_prediction_format(path):
    """Vérifie que y_pred.csv respecte le format"""
    pass
