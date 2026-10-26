import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze("columns")

# Charger les meilleurs paramètres
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Créer le modèle avec les meilleurs paramètres
model = RandomForestRegressor(
    **best_params,
    random_state=42
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Entraînement terminé avec succès !")