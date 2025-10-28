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
    
    ordinal_features = ['sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    
    for col in ordinal_features:
        if col in X.columns:
            X[col] = X[col].map(ordinal_mapping)
    
    return X

def one_hot_encoding(X:pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X, columns=['profession'], drop_first=True)

def scale_features(
    X: pd.DataFrame, 
    scaler: StandardScaler = None, 
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

def preprocess_pipeline(
    X: pd.DataFrame, 
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
    add_features: bool = True
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Pipeline complet de preprocessing pour les données Smurf.
    
    Étapes :
    1. Suppression de la colonne 'image_name'
    2. Plafonnement des outliers
    3. Gestion des valeurs manquantes
    4. Feature engineering (BMI)
    5. Encodage ordinal (sarsaparilla, liquor, donuts)
    6. One-hot encoding (profession)
    7. Standardisation
    
    Args:
        X (pd.DataFrame): DataFrame brut
        scaler (StandardScaler): Scaler pré-fitté (None pour train)
        fit_scaler (bool): Si True, fit le scaler. False pour test/unlabeled.
        add_features (bool): Si True, ajoute BMI et autres features engineerées
    
    Returns:
        tuple[pd.DataFrame, StandardScaler]: DataFrame prétraité + scaler
    
    Example:
        >>> # Train
        >>> X_train_prep, scaler = preprocess_pipeline(X_train, fit_scaler=True)
        >>> 
        >>> # Test (même preprocessing, même scaler)
        >>> X_test_prep, _ = preprocess_pipeline(X_test, scaler=scaler, fit_scaler=False)
    """
    X = X.copy()
    if 'image_name' in X.columns:
        X = X.drop('image_name', axis=1)
    
    thresholds = {
        'blood pressure': 160,  # À ajuster par rapport à l'EDA
    }
    X = cap_outliers(X, thresholds)
    X = handle_missing_values(X, strategy='median')
    
    if add_features:
        X = add_bmi(X)
    
    X = encode_ordinal_features(X)
    X = one_hot_encoding(X)
    X, scaler = scale_features(X, scaler=scaler, fit=fit_scaler)
    
    return X, scaler