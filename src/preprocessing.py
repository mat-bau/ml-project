"""
Module: preprocessing.py
Description: Ensure the data is preprocessed before training our model
    -> Data cleaning : addressing anomalies (outliers, missing values) -> Data transformation : normalization and rescaling
    -> Structural operations: feature engeneering
Course: LELEC2870
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def cap_outliers(X: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Cap outliers in X's feature.
    
    Args:
        X (pd.DataFrame): DataFrame à nettoyer
        thresholds (dict): Dictionnaire {feature: max_value}
            Ex: {'blood pressure': 180, 'age': 250}
    
    Returns:
        pd.DataFrame: DataFrame avec outliers plafonnés
        
    Example:
        >>> thresholds = {'blood pressure': 180, 'age': 250}
        >>> X_clean = cap_outliers(X_train, thresholds)
    """
    X = X.copy()
    
    for feature, max_val in thresholds.items():
        if feature in X.columns:
            X.loc[X[feature] > max_val, feature] = max_val
    
    return X

def handle_missing_values(X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute les valeurs manquantes.
    
    Args:
        X: DataFrame avec valeurs manquantes
        strategy: 'median', 'mean', ou 'mode'
    
    Returns:
        DataFrame sans valeurs manquantes
    """
    X = X.copy()
    
    if strategy == 'median':
        X = X.fillna(X.median(numeric_only=True))
    elif strategy == 'mean':
        X = X.fillna(X.mean(numeric_only=True))
    elif strategy == 'mode':
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0])
    
    return X

def add_bmi(X:pd.DataFrame) -> pd.DataFrame:
    """Adds BMI = weight / (height/100)^2"""
    X = X.copy()
    X['bmi'] = X['weight'] / (X['height'] / 100) ** 2
    return X

def encode_ordinal_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les features ordinales (very low → 0, ..., very high → 4).
    
    Args:
        X (pd.DataFrame): DataFrame avec features ordinales
    
    Returns:
        pd.DataFrame: DataFrame avec features encodées
    
    Notes:
        Utilise un mapping manuel pour avoir un contrôle total sur l'ordre.
    """
    X = X.copy()
    
    ordinal_mapping = {
        'Very low': 0,
        'Low': 1,
        'Moderate': 2,
        'High': 3,
        'Very high': 4
    }
    
    ordinal_features = ['sarsaparilla', 'smurfberry liquor', 'smurfin donuts', 'profession']
    
    for col in ordinal_features:
        if col in X.columns:
            X[col] = X[col].map(ordinal_mapping)
    
    return X

def one_hot_encoding(X:pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X, columns=['sarsaparilla', 'smurfberry liquor', 'smurfin donuts', 'profession'], drop_first=True)

def scale_features(
    X: pd.DataFrame, 
    scaler: StandardScaler = StandardScaler(), 
    fit: bool = True
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numerical feature.
    
    Args:
        X (pd.DataFrame): DataFrame à standardiser
        scaler (StandardScaler): Scaler pré-fitté (pour test/unlabeled)
        fit (bool): Si True, fit le scaler sur X. Si False, utilise scaler existant.
    
    Returns:
        tuple[pd.DataFrame, StandardScaler]: DataFrame standardisé + scaler utilisé
    
    Example:
        >>> # Train
        >>> X_train_scaled, scaler = scale_features(X_train, fit=True)
        >>> # Test (utilise le même scaler)
        >>> X_test_scaled, _ = scale_features(X_test, scaler=scaler, fit=False)
    
    Warning:
        TOUJOURS utiliser fit=True sur train et fit=False sur test pour éviter 
        le data leakage !!
    """
    X = X.copy()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist() # selectione les colonnes numeriques
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X, scaler


"""
def remove_image(X:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "img_filename" in X.columns:
        img_filename_column = X["img_filename"]
        X_encoded    = X.drop(columns = ["img_filename"])
        return X_encoded, img_filename_column
    return X, pd.Series(dtype=object)
"""

def preprocess_pipeline(X, scaler=None, fit_scaler=True):
    """
    Nettoie et standardise les données d'entrée.
    Garde le même scaler entre train, validation, test et unlabeled.
    """
    X_copy = X.copy()

    # Remplacement des valeurs manquantes
    X_copy = X_copy.fillna(X_copy.median(numeric_only=True))

    # Supprimer les colonnes non numériques (s’il y en a encore)
    X_copy = X_copy.select_dtypes(include=[np.number])

    # Fit ou réutilise le scaler
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_copy)
    else:
        X_scaled = scaler.transform(X_copy)

    # Retourne sous forme DataFrame avec les mêmes noms de colonnes
    return pd.DataFrame(X_scaled, columns=X_copy.columns, index=X_copy.index), scaler


# Preprocessing SANS scaling pour Random Forest
def preprocess_pipeline_no_scaling(X, fit_scaler=True, add_features:bool = True):
    """Version sans scaling pour les modèles basés sur arbres."""
    X = X.copy()
    
    if 'img_filename' in X.columns:
        X = X.drop('img_filename', axis=1)
    
    thresholds = {
        'blood pressure': 160,  # À ajuster par rapport à l'EDA
    }
    X = cap_outliers(X, thresholds)
    X = handle_missing_values(X, strategy='median')
    
    if add_features:
        X = add_bmi(X)
    
    X = encode_ordinal_features(X)
    X = one_hot_encoding(X)
    
    return X

def preprocess_data(X, scaler=None, fit_scaler=True):
    """Pipeline de preprocessing complet"""
    X = X.copy()
    
    # Outliers
    if 'blood pressure' in X.columns:
        X.loc[X['blood pressure'] > 160, 'blood pressure'] = 160
    
    # Feature engineering: BMI
    X['bmi'] = X['weight'] / (X['height'] / 100) ** 2
    
    # Encodage ordinal
    ordinal_mapping = {'Very low': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very high': 4}
    ordinal_cols = ['sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    for col in ordinal_cols:
        if col in X.columns:
            X[col] = X[col].map(ordinal_mapping)
    
    # One-hot encoding
    X = pd.get_dummies(X, columns=['profession'], drop_first=True)
    
    # Imputation
    num_cols = X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    X[num_cols] = imputer.fit_transform(X[num_cols])
    
    # Scaling
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        X[num_cols] = scaler.fit_transform(X[num_cols])
    else:
        X[num_cols] = scaler.transform(X[num_cols])
    
    return X, scaler