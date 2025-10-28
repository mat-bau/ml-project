"""
Module: utils.py
Description: Set of functions to simplify things
Course: LELEC2870
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import joblib


def load_preprocessing_objects(models_dir: str = "../results/models") -> Dict[str, object]:
    """
    Load a dictionnary containing main objects computed with the preprocessing phase.

    Args:
        models_dir (str): Path to the directory where objects are saved
    Returns:
        (Dict[str, object]) : 
            - scaler = mean and std
            - ordinal_ordering used for mapping ordinal features
            - feature_names
    Example:
        >>> prep_objects = load_preprocessing_objects('../results/models')
    """
    return {
        'scaler': joblib.load(f'{models_dir}/scaler.pkl'),
        'ordinal_ordering': joblib.load(f'{models_dir}/ordinal_ordering.pkl'),
        'feature_names': joblib.load(f'{models_dir}/feature_names.pkl'),
    }

def save_predictions(y_pred: pd.DataFrame, path: str ='predictions/y_pred.csv') -> None:
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

def verify_prediction_format(predictionFile_path:str):
    df = pd.read_csv(predictionFile_path)
    df_expected = pd.read_csv("../data/data_labeled/y_train.csv")

    # 1 seule colonne
    assert df_expected.columns.tolist() == df.columns.tolist(), (
        "Format invalide : il faut une seul colonne"
    )

    # pas d'entete


    # 1 chiffre par ligne



    return df
    
