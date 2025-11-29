import pandas as pd

# Lire le fichier y_pred.csv (sans header)
y_pred = pd.read_csv("y_pred.csv", header=None)

# Remplacer les valeurs négatives par 0
y_pred[0] = y_pred[0].clip(lower=0)

# Sauvegarder le fichier corrigé
y_pred.to_csv("y_pred_corrected.csv", index=False, header=False)

print("✅ Fichier 'y_pred_corrected.csv' créé avec les valeurs négatives remplacées par 0.")
