import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# --- 1. Charger les données ---
path_train = "data/data_labeled/X_train.csv"
path_ytrain = "data/data_labeled/y_train.csv"
path_test = "data/data_labeled/X_test.csv"
path_ytest = "data/data_labeled/y_test.csv"
path_unlabeled = "data/data_unlabeled/X.csv"

X_train = pd.read_csv(path_train)
y_train = pd.read_csv(path_ytrain, header=None).values.ravel()
X_test = pd.read_csv(path_test)
y_test = pd.read_csv(path_ytest, header=None).values.ravel()

# Retirer la colonne d'image (inutile pour part 1)
X_train = X_train.drop(columns=['img_filename'])
X_test = X_test.drop(columns=['img_filename'])

# --- 2. Identifier les types de variables ---
num_features = [
    "age", "blood pressure", "calcium", "cholesterol", "hemoglobin",
    "height", "potassium", "vitamin D", "weight"
]
cat_features = [
    "profession", "sarsaparilla", "smurfberry liquor", "smurfin donuts"
]

# --- 3. Préprocessing pipeline ---
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# --- 4. Modèle linéaire ---
model = Ridge(alpha=1.0)  # ridge = régression linéaire régularisée

# --- 5. Pipeline complet ---
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])

# --- 6. Entraînement ---
pipeline.fit(X_train, y_train)

# --- 7. Évaluation sur le test set ---
y_pred_test = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"✅ RMSE sur test set = {rmse:.4f}")

# --- 8. Générer prédictions sur données non labellisées ---
X_unlabeled = pd.read_csv(path_unlabeled)
X_unlabeled = X_unlabeled.drop(columns=['img_filename'])
y_pred_unlabeled = pipeline.predict(X_unlabeled)

# --- 9. Sauvegarde du fichier final ---
np.savetxt("y_pred.csv", y_pred_unlabeled, fmt="%.6f")
print("📁 Fichier y_pred.csv généré avec succès.")
