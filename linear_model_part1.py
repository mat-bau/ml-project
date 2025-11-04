# ============================================
# CODE DU QUEL IL FAUT S'INSPIRER POUR CHANGER LE NOTEBOOK
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# -----------------------------
# 1. Load data
# -----------------------------
PATH_TRAIN = "data/data_labeled/X_train.csv"
PATH_YTRAIN = "data/data_labeled/y_train.csv"
PATH_TEST = "data/data_labeled/X_test.csv"
PATH_YTEST = "data/data_labeled/y_test.csv"
PATH_UNLABELED = "data/data_unlabeled/X.csv"

X_train = pd.read_csv(PATH_TRAIN)
y_train = pd.read_csv(PATH_YTRAIN, header=None).values.ravel()
X_test = pd.read_csv(PATH_TEST)
y_test = pd.read_csv(PATH_YTEST, header=None).values.ravel()
X_unlabeled = pd.read_csv(PATH_UNLABELED)

# Drop the image column (not used for part 1)
for df in [X_train, X_test, X_unlabeled]:
    if "img_filename" in df.columns:
        df.drop(columns=["img_filename"], inplace=True)

# -----------------------------
# 2. Define feature types
# -----------------------------
num_features = [
    "age", "blood pressure", "calcium", "cholesterol", "hemoglobin",
    "height", "potassium", "vitamin D", "weight"
]
cat_features = [
    "profession", "sarsaparilla", "smurfberry liquor", "smurfin donuts"
]

# -----------------------------
# 3. Preprocessing pipeline
# -----------------------------
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

# -----------------------------
# 4. Split train/validation set
# -----------------------------
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# 5. Define models to compare
# -----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet()
}

# -----------------------------
# 6. Evaluate models with cross-validation
# -----------------------------
cv_results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    # scoring='neg_root_mean_squared_error' renvoie -RMSE, donc on prend le signe inverse
    scores = cross_val_score(pipeline, X_train_split, y_train_split,
                             scoring="neg_root_mean_squared_error", cv=5)
    rmse_mean = -scores.mean()
    rmse_std = scores.std()
    cv_results[name] = (rmse_mean, rmse_std)
    print(f"{name:<15} | RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")

# -----------------------------
# 7. Select best model
# -----------------------------
best_model_name = min(cv_results, key=lambda k: cv_results[k][0])
print(f"\n✅ Best base model: {best_model_name} (RMSE={cv_results[best_model_name][0]:.4f})")

# -----------------------------
# 8. Hyperparameter tuning (Ridge, Lasso, ElasticNet)
# -----------------------------
tuned_models = {
    "Ridge": {"model__alpha": [0.01, 0.1, 1, 10, 100, 600]},
    "Lasso": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet": {
        "model__alpha": [0.001, 0.01, 0.1, 1],
        "model__l1_ratio": [0.2, 0.5, 0.8]
    }
}

if best_model_name in tuned_models:
    print(f"\n🔍 Fine-tuning {best_model_name} with GridSearchCV...")
    grid = GridSearchCV(
        Pipeline([("preprocessor", preprocessor),
                  ("model", models[best_model_name])]),
        tuned_models[best_model_name],
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train_split, y_train_split)
    best_model = grid.best_estimator_
    best_rmse = -grid.best_score_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV RMSE: {best_rmse:.4f}")
else:
    # For LinearRegression
    best_model = Pipeline([("preprocessor", preprocessor),
                           ("model", models[best_model_name])])
    best_model.fit(X_train_split, y_train_split)
    best_rmse = cv_results[best_model_name][0]

# -----------------------------
# 9. Evaluate on validation + test
# -----------------------------
best_model.fit(X_train_split, y_train_split)
y_val_pred = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"\n💡 RMSE on validation set = {val_rmse:.4f}")

y_test_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"📊 RMSE on test set = {test_rmse:.4f}")

# -----------------------------
# 10. Final predictions on unlabeled data
# -----------------------------
y_pred_unlabeled = best_model.predict(X_unlabeled)
np.savetxt("y_pred.csv", y_pred_unlabeled, fmt="%.6f")
print("\n📁 File y_pred.csv successfully created.")
